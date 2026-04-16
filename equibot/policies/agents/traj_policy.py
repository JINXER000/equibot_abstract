import copy
import hydra
import torch
from torch import nn
import torch.nn.functional as F

from equibot.policies.vision.sim3_encoder import SIM3Vec4Latent
from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.equivariant_diffusion.conditional_unet1d import VecConditionalUnet1D
from equibot.policies.utils.equivariant_diffusion.unconditional_mlp import UnconditionalMLP
import numpy as np

from equibot.policies.utils.misc import to_torch, \
    convert_trans_to_vec, convert_vec_to_trans, ActionSlice,\
    rotation_6d_to_matrix, geodestDist


    
class TrajPolicy(nn.Module):
    def __init__(self, cfg, device="cpu"):
        nn.Module.__init__(self)
        self.obs_mode = cfg.model.obs_mode
        self.ac_mode = cfg.model.ac_mode
        self.use_torch_compile = cfg.model.use_torch_compile
        self.device = device

        # |o|o|                             observations: 2
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        # | |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        self.pred_horizon = cfg.model.pred_horizon
        self.obs_horizon = cfg.model.obs_horizon
        self.action_horizon = cfg.model.ac_horizon
        self.symb_mask = cfg.data.dataset.symb_mask
        has_eff_list = cfg.data.dataset.has_eff_list
        self.has_eff_dict = {'left': False, 'right': False}
        hand_sides = ['left', 'right']
        for i in range(len(hand_sides)):
            self.has_eff_dict[hand_sides[i]] = has_eff_list[i]

        if hasattr(cfg.model, "num_diffusion_iters"):
            self.num_diffusion_iters = cfg.model.num_diffusion_iters
        else:
            self.num_diffusion_iters = cfg.model.noise_scheduler.num_train_timesteps

        self.left_encoder = SIM3Vec4Latent(**cfg.model.encoder) # hidden_dim = 32
        self.right_encoder = SIM3Vec4Latent(**cfg.model.encoder) # hidden_dim = 32

        self.encoder_out_dim = cfg.model.encoder.c_dim

        # self.mask_type = self.conclude_masks()

        # self.num_eef = cfg.env.num_eef
        self.dof = cfg.env.dof # 6
        self.eef_dims  = {'left': 3, 'right':3} # xyz, dir1, dir2
        for side in hand_sides:
            if self.has_eff_dict[side]:
                self.eef_dims[side] = 6
        self.num_eef = cfg.env.num_eef

        self.obs_dim = self.encoder_out_dim

        self.left_noise_pred_net = VecConditionalUnet1D(
            input_dim=self.eef_dims['left'],  ## vec dim, rot is 2, xyz is 1
            cond_dim=self.obs_dim* self.obs_horizon,
            scalar_cond_dim=0,
            scalar_input_dim= 1,
            diffusion_step_embed_dim=self.obs_dim* self.obs_horizon,
            cond_predict_scale=True,  ## in Fila, do AX+B instead of x+B
        )
        self.right_noise_pred_net = VecConditionalUnet1D(
            input_dim=self.eef_dims['right'],
            cond_dim=self.obs_dim* self.obs_horizon,
            scalar_cond_dim=0,
            scalar_input_dim= 1,
            diffusion_step_embed_dim=self.obs_dim* self.obs_horizon,
            cond_predict_scale=True,
        )
        joint_scalar_dims = self.dof * self.num_eef  

        self.jpose_noise_pred_net = UnconditionalMLP(
            input_dim= joint_scalar_dims,
            diffusion_step_embed_dim=self.obs_dim* self.obs_horizon,
        )

        self.nets = nn.ModuleDict(
            {"left_encoder": self.left_encoder, \
                "right_encoder": self.right_encoder, \
             "left_noise_pred_net": self.left_noise_pred_net,\
             "right_noise_pred_net": self.right_noise_pred_net,\
                "jpose_noise_pred_net": self.jpose_noise_pred_net}
        )
        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        self._init_torch_compile()

        self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler)

        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized paraGen Policy with {num_parameters} parameters")


    def _init_torch_compile(self):
        if self.use_torch_compile:
            self.left_encoder_handle = torch.compile(self.left_encoder)
            self.right_encoder_handle = torch.compile(self.right_encoder)
            self.left_noise_pred_net_handle = torch.compile(self.left_noise_pred_net)
            self.right_noise_pred_net_handle = torch.compile(self.right_noise_pred_net)
        else:
            self.left_encoder_handle = self.left_encoder
            self.right_encoder_handle = self.right_encoder
            self.left_noise_pred_net_handle = self.left_noise_pred_net
            self.right_noise_pred_net_handle = self.right_noise_pred_net

    # def _convert_jpose_to_vec(self, jpose, batch=None):
    #     # input: (B, 1, E , dof); output: (B, 1, ac_dim, 3) 
    #     # jpose = jpose.reshape(jpose.shape[0], jpose.shape[1],  -1, 3)
    #     jpose = jpose.reshape(jpose.shape[0], -1,  self.dof * self.num_eef)
    #     return jpose
    
    def step_ema(self):
        self.ema.step(self.nets)

    def normalize_from_key(self, key, data):
        return self.all_normalizers[key].normalize(data)
    
    def unnormalize_from_key(self, key, data):
        return self.all_normalizers[key].unnormalize(data)

    def recover_grasp(self, grasp_batch, scale, center, key):
        side = key.split('_')[0]
        has_eff = self.has_eff_dict[side]

        # reshape dim to B,  (3 or 6) , 3
        # grasp_batch = torch.mean(grasp_batch, dim=1, keepdim=True)
        scale = torch.mean(scale, dim=2, keepdim=True)
        center = torch.mean(center, dim=2, keepdim=True)

        batch_size = grasp_batch.shape[0]

        ##### grasp processing
        if has_eff == False:
            grasp_xyz = grasp_batch[:, :,0, :].reshape(batch_size, -1, 1, 3)
        else:
            grasp_xyz = grasp_batch[:, :, :2, :].reshape(batch_size, -1, 2, 3)

        # add back the offset
        grasp_xyz = grasp_xyz *scale + center

        # un-normalize
        unnormed_grasp_xyz = (
                    self.unnormalize_from_key(key, grasp_xyz)
                )
        
        ##### rotation processing 
        if has_eff == False:
            rot6d_batch = grasp_batch[:, :, 1: , :].reshape(batch_size, -1, 1, 6)
        else:
            # rot6d_batch = grasp_batch[:, :, 2:, :].reshape(-1, 1, 2, 6)
            grasp_dir1 = grasp_batch[:, :, 2:4, :]
            grasp_dir2 = grasp_batch[:, :, 4:6, :]
            rot6d_batch = torch.cat((grasp_dir1, grasp_dir2), dim=-1)
            assert rot6d_batch.shape[-1] == 6

        trans_batch = convert_vec_to_trans(rot6d_batch, unnormed_grasp_xyz, has_eff=has_eff)

        trans_batch = trans_batch.detach().cpu().numpy()

        return trans_batch, unnormed_grasp_xyz, rot6d_batch

    def recover_jpose(self, jpose_batch, key):
        # squeeze the pred horizon to 1
        jpose_action = jpose_batch.reshape(-1,  self.num_eef, self.dof)
        unnormed_joint = (
                    self.unnormalize_from_key(key, jpose_action)
                    .detach()
                    .cpu()
                    .numpy()
                )
        
        return unnormed_joint

    def recover_gripper(self, gripper_batch, key):
        unnormed_gripper = (
                    self.unnormalize_from_key(key, gripper_batch)
                    .detach()
                    .cpu()
                    .numpy()
                )
        return unnormed_gripper

    def proc_pc(self, pc, key, ema_nets = None):
        pc = self.normalize_from_key(key, pc)
        batch_size = pc.shape[0]
        side = key.split('_')[0]
        ## in training
        if ema_nets is None:
            encoder_handle = self.left_encoder if side == 'left' else self.right_encoder
            feat_dict = encoder_handle(pc, target_norm=self.all_normalizers[key+'_scale'])
        else: # in inference
            feat_dict = ema_nets[side+"_encoder"](pc, ret_perpoint_feat=False, target_norm=self.all_normalizers[key+'_scale'])
        
        center = (
            feat_dict["center"].reshape(batch_size, self.obs_horizon, 1, 3)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        )
        scale = feat_dict["scale"].reshape(batch_size, self.obs_horizon, 1, 1)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        pc_feat = feat_dict["so3"]  
        obs_cond_vec = pc_feat.reshape(batch_size, -1, 3)
        return obs_cond_vec, center, scale

    def proc_grasp(self, grasp_pose, key, center, scale):
        side = key.split('_')[0]
        has_eff = self.has_eff_dict[side]
        grasp_xyz_raw, grasp_dir1, grasp_dir2 = convert_trans_to_vec(grasp_pose, has_eff=has_eff)
        grasp_xyz = self.normalize_from_key(key, grasp_xyz_raw)
        grasp_xyz = (grasp_xyz - center)/scale
        gt_grasp_z = torch.cat([grasp_xyz, grasp_dir1, grasp_dir2], dim=-2)

        # ## check same
        # trans_batch, unnormed_grasp_xyz, rot6d_batch = self.recover_grasp(gt_grasp_z, scale, center, key)
        # trans_batch_ts = torch.tensor(trans_batch).to(self.device)
        # error = nn.functional.mse_loss(trans_batch_ts, grasp_pose)
        # assert error < 1e-6
        return gt_grasp_z

    def proc_gripper(self, raw_gripper, key):
        gripper_action = self.normalize_from_key(key, raw_gripper)
        return gripper_action
    
    def proc_jpose(self, jpose, key):
        
        jpose_n = self.normalize_from_key(key, jpose)
        # jpose_vec = self._convert_jpose_to_vec(jpose_n)
        jpose_vec = jpose_n.reshape(jpose_n.shape[0], -1,  self.dof * self.num_eef)
        return jpose_vec
    
    def pred_bimanual_jposes(self, batch_size, gt_batch = None):

        ema_nets = self.ema.averaged_model

        initial_noise_scale = 1
        noisy_jpose = torch.randn((batch_size,   self.num_eef*self.dof)).to(self.device) * initial_noise_scale

        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        curr_action = {"dual_jpose": noisy_jpose}

        ####### inverse diffusion step
        for k in self.noise_scheduler.timesteps:
            new_action = {"dual_jpose": None}

            scalar_noise_pred = ema_nets["jpose_noise_pred_net"](\
                sample=curr_action["dual_jpose"],
                timesteps = k,
            )
            new_action['dual_jpose'] = self.noise_scheduler.step(
                model_output=scalar_noise_pred, timestep=k, sample=curr_action["dual_jpose"]
            ).prev_sample

            curr_action = new_action

        unnormed_joint = self.recover_jpose(curr_action['dual_jpose'], key='dual_jpose')
        unnormed_joint = torch.tensor(unnormed_joint).to(self.device)   

        action_dict = {}
        eval_metrics = {}
        if batch_size ==1:
            action_dict['dual_jpose'] = unnormed_joint.reshape(self.num_eef, self.dof)
        else:
            gt_joint = gt_batch["dual_jpose"]
            joint_mse = torch.nn.functional.mse_loss(unnormed_joint, gt_joint)
            eval_metrics["dual_joint_mse"] = joint_mse

        return action_dict, eval_metrics
    
    def pred_unimanual_traj(self, side, agent_obs, gt_batch = None):
        pc_data = agent_obs[side + '_pc']
        batch_size =  pc_data.shape[0]
        pc_data = pc_data.repeat(1, self.obs_horizon, 1, 1)

        ema_nets = self.ema.averaged_model

        obs_vec, center, scale = self.proc_pc(pc_data, side + '_pc', ema_nets = ema_nets)

        ##### start denoising #####

        initial_noise_scale = 1
        noisy_eef_xt = torch.randn((batch_size, self.pred_horizon, self.eef_dims[side], 3)).to(self.device)\
        * initial_noise_scale

        noisy_gripper = torch.randn((batch_size, self.pred_horizon, 1)).to(self.device) * initial_noise_scale

        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        curr_action = {side + "_grasp": noisy_eef_xt, 
            side + "_gripper": noisy_gripper}
        
         ####### inverse diffusion step
        for k in self.noise_scheduler.timesteps:

            new_action = {side + "_grasp": None, side+ '_gripper': None }

            vec_noise_pred, gripper_noise_pred = ema_nets[side+"_noise_pred_net"](\
                sample=curr_action[side+ "_grasp"],
                timestep = k,
                scalar_sample = curr_action[side + "_gripper"], 
                cond= obs_vec,
                scalar_cond=None,
            )
            new_action[side+ "_grasp"] = self.noise_scheduler.step(
                model_output=vec_noise_pred, timestep=k, sample=curr_action[side+ "_grasp"]
            ).prev_sample

            new_action[side + "_gripper"] = self.noise_scheduler.step(
                model_output=gripper_noise_pred, timestep=k, sample=curr_action[side + "_gripper"]
            ).prev_sample

            curr_action = new_action

        ### recover the grasp eef pose
        has_eff = self.has_eff_dict[side]
        ## predicted values
        trans_batch, unnormed_grasp_xyz, rot6d_batch = self.recover_grasp(\
            curr_action[side+'_grasp'], scale, center, key=side+'_grasp')
        assert trans_batch.shape[3] == 4

        ## recover gripper
        gripper_batch = self.recover_gripper(curr_action[side+'_gripper'], key=side+'_gripper')

        ## update action dict 
        action_dict = {}
        eval_metrics = {}
        if batch_size ==1:
            action_dict[side+'_grasp'] = trans_batch[0]
            action_dict[side+'_gripper'] = gripper_batch[0]
        ## calc metrics if in training
        else:
            gt_grasp_xyz, gt_dir1, gt_dir2 = convert_trans_to_vec(gt_batch[side+"_grasp"], has_eff=has_eff)
            # gt_grasp_rot6d = torch.cat([gt_dir1, gt_dir2], dim=-1)

            # xyz_mse = torch.nn.functional.mse_loss(unnormed_grasp_xyz, gt_grasp_xyz)
            # rot_mse = torch.nn.functional.mse_loss(rot6d_batch, gt_grasp_rot6d)
            # eval_metrics[side+"_xyz_mse"] = xyz_mse
            # eval_metrics[side+"_rot_mse"] = rot_mse
            xyz_l1 = torch.nn.functional.l1_loss(unnormed_grasp_xyz, gt_grasp_xyz)
            eval_metrics[f"{side}:xyz_l1"] = xyz_l1

            gt_eefpos_rot6d = torch.cat([gt_dir1, gt_dir2], dim=-1)
            gt_Rs = rotation_6d_to_matrix(gt_eefpos_rot6d)
            pred_Rs = rotation_6d_to_matrix(rot6d_batch)
            diff_theta = geodestDist(gt_Rs, pred_Rs).mean()
            eval_metrics[f"{side}:rot_diff"] = diff_theta * 180 / torch.pi
        return action_dict, eval_metrics

    def forward(self, batch, history_bid=-1):
        ###### preprocess data #######
        batch = to_torch(batch, self.device)
        batch_size = batch['dual_jpose'].shape[0]

        action_dict_all = {}
        eval_metrics_all = {}
        agent_obs = {}
        for side in ["left", "right"]:
            agent_obs[side + '_pc'] = batch[side + '_pc']
            action_dict, eval_metrics = self.pred_unimanual_traj(side, batch, gt_batch=batch)
            action_dict_all.update(action_dict)
            eval_metrics_all.update(eval_metrics)

        action_dict, eval_metrics = self.pred_bimanual_jposes(batch_size=batch_size, gt_batch=batch)
        action_dict_all.update(action_dict)
        eval_metrics_all.update(eval_metrics)

        denoise_history = []
        if history_bid >= 0 and len(action_dict_all) > 0:
            ## in traj mode, we do not visulize the history. Instead, we visualize the final action
            traj_len = self.pred_horizon
            for i in range(traj_len):
                action_slice = ActionSlice(mode="separated")
                for side in ["left", "right"]:
                    action_slice.update(side+'_grasp', action_dict_all[side+'_grasp'][i])

                    action_slice.update(side+'_gripper', action_dict_all[side+'_gripper'][i])

                action_slice.update('dual_jpose', action_dict_all['dual_jpose'].reshape(-1))

                denoise_history.append(action_slice)
        return action_dict_all, eval_metrics_all, denoise_history

