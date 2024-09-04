import copy
import hydra
import torch
from torch import nn
import numpy as np

from equibot.policies.vision.sim3_encoder import SIM3Vec4Latent
from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.equivariant_diffusion.conditional_unet1d import VecConditionalUnet1D

from equibot.policies.utils.misc import rotation_6d_to_matrix, matrix_to_rotation_6d

# from torch.utils.tensorboard import SummaryWriter

class ALOHAPolicy(nn.Module):
    # TODO: figure out the dimensions!
    def __init__(self, cfg, device = "cpu"):
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
        self.has_eff = (cfg.data.dataset.dataset_type == 'hdf5_predeff')

        if hasattr(cfg.model, "num_diffusion_iters"):
            self.num_diffusion_iters = cfg.model.num_diffusion_iters
        else:
            self.num_diffusion_iters = cfg.model.noise_scheduler.num_train_timesteps

        if self.obs_mode.startswith("pc"):
            self.encoder = SIM3Vec4Latent(**cfg.model.encoder) # hidden_dim = 32
        else:
            self.encoder = None
        self.encoder_out_dim = cfg.model.encoder.c_dim

        # self.num_eef = cfg.env.num_eef
        self.dof = cfg.env.dof # 6
        if self.has_eff == False:
            self.eef_dim = 3 # xyz, dir1, dir2
        else:
            self.eef_dim = 6
        self.num_eef = len([x for x in self.symb_mask[:2] if x != 'None'])
        num_scalar_dims = self.dof * self.num_eef # joint pose

        self.obs_dim = self.encoder_out_dim
        self.noise_pred_net = VecConditionalUnet1D(
            input_dim=self.eef_dim,
            cond_dim=self.obs_dim* self.obs_horizon,
            scalar_cond_dim=0,
            scalar_input_dim= num_scalar_dims,
            diffusion_step_embed_dim=self.obs_dim* self.obs_horizon,
            cond_predict_scale=True,
        )

        self.nets = nn.ModuleDict(
            {"encoder": self.encoder, "noise_pred_net": self.noise_pred_net}
        )
        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        self._init_torch_compile()

        self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler)

        # self.writer = SummaryWriter()
        self.mask_type = self.conclude_masks()

        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized paraGen Policy with {num_parameters} parameters")

    def _init_torch_compile(self):
        if self.use_torch_compile:
            self.encoder_handle = torch.compile(self.encoder)
            self.noise_pred_net_handle = torch.compile(self.noise_pred_net)
        else:
            self.encoder_handle = self.encoder
            self.noise_pred_net_handle = self.noise_pred_net

    def _convert_jpose_to_vec(self, jpose, batch=None):
        # input: (B, 1, E , dof); output: (B, 1, ac_dim, 3) 
        # jpose = jpose.reshape(jpose.shape[0], jpose.shape[1],  -1, 3)
        jpose = jpose.reshape(jpose.shape[0], -1,  self.dof * self.num_eef)
        return jpose
    
    # def _convert_grasp_to_vec(self, grasp, batch = None):
    #     assert grasp.shape[-1] == 9
    #     eef_pose = grasp[:, :, :, :3]
    #     dir1 = grasp[:, :, :, 3:6]
    #     dir2 = grasp[:, :, :, 6:9]

    #     return eef_pose, dir1, dir2
    
    def _convert_trans_to_vec(self, grasp_trans_arr):
        batch_size, horizon, _, _ = grasp_trans_arr.shape
        grasp_xyz = grasp_trans_arr[:, :, :3, 3].reshape(batch_size, horizon, 1, 3)  # B, H, 1, 3
        grasp_rot =  grasp_trans_arr[:, :, :3, :3].reshape(-1, 3, 3) # B*H, 3, 3
        rot6d = matrix_to_rotation_6d(grasp_rot) # B*H, 6
        rot_dir1 = rot6d[:, :3].reshape(batch_size, horizon, 1, 3)
        rot_dir2 = rot6d[:, 3:].reshape(batch_size, horizon, 1, 3)

        if self.has_eff:
            eff_grasp_xyz = grasp_trans_arr[:, :, 4:7, 3].reshape(batch_size, horizon, 1, 3)
            eff_grasp_rot =  grasp_trans_arr[:, :, 4:7, :3].reshape(-1, 3, 3)
            eff_rot6d = matrix_to_rotation_6d(eff_grasp_rot)
            eff_rot_dir1 = eff_rot6d[:, :3].reshape(batch_size, horizon, 1, 3)
            eff_rot_dir2 = eff_rot6d[:, 3:].reshape(batch_size, horizon, 1, 3)

            # combine pred and eff tensors
            grasp_xyz = torch.cat((grasp_xyz, eff_grasp_xyz), dim=2)
            rot_dir1 = torch.cat((rot_dir1, eff_rot_dir1), dim=2)
            rot_dir2 = torch.cat((rot_dir2, eff_rot_dir2), dim=2)

        return grasp_xyz, rot_dir1, rot_dir2
    
    def _convert_vec_to_trans(self, rot6d_batch, unnormed_grasp_xyz):
        batch_size, horizon, grasp_num, vec_dim = rot6d_batch.shape
        if self.has_eff == False:
            assert grasp_num == 1
            rot6d_batch = rot6d_batch.reshape(-1, 6)
            rotation_mat_ts = rotation_6d_to_matrix(rot6d_batch)
            rotation_mat = rotation_mat_ts

            trans_mat_batch = torch.zeros((batch_size * horizon, 4, 4), device=rot6d_batch.device)
            trans_mat_batch[:, :3, :3] = rotation_mat
            trans_mat_batch[:, :3, 3] = unnormed_grasp_xyz.reshape(-1, 3)
            trans_mat_batch[:, 3, 3] = 1

            trans_mat_batch = trans_mat_batch.reshape(batch_size, horizon, 4, 4)
        else:
            assert grasp_num == 2
            unnormed_grasp_xyz_pre = unnormed_grasp_xyz[:, :, 0, :].reshape(-1, 3)
            rot6d_batch_pre = rot6d_batch[:, :, 0, :].reshape(-1, 6)
            rotation_mat_pre = rotation_6d_to_matrix(rot6d_batch_pre)
            unnormed_grasp_xyz_eff = unnormed_grasp_xyz[:, :, 1, :].reshape(-1, 3)
            rot6d_batch_eff = rot6d_batch[:, :, 1, :].reshape(-1, 6)
            rotation_mat_eff = rotation_6d_to_matrix(rot6d_batch_eff)

            trans_mat_batch = torch.zeros((batch_size * horizon, 8, 4), device=rot6d_batch.device)
            trans_mat_batch[:, :3, :3] = rotation_mat_pre
            trans_mat_batch[:, :3, 3] = unnormed_grasp_xyz_pre.reshape(-1, 3)
            trans_mat_batch[:, 3, 3] = 1
            trans_mat_batch[:, 4:7, :3] = rotation_mat_eff
            trans_mat_batch[:, 4:7, 3] = unnormed_grasp_xyz_eff.reshape(-1, 3)
            trans_mat_batch[:, 7, 3] = 1

        trans_mat_batch = trans_mat_batch.reshape(batch_size, horizon, -1, 4)
        return trans_mat_batch


    def step_ema(self):
        self.ema.step(self.nets)


    def recover_grasp(self, grasp_batch, scale, center):
        # reshape dim to B,  (3 or 6) , 3
        grasp_batch = torch.mean(grasp_batch, dim=1, keepdim=True)
        scale = torch.mean(scale, dim=1, keepdim=True)
        center = torch.mean(center, dim=1, keepdim=True)

        ##### grasp processing
        if self.has_eff == False:
            grasp_xyz = grasp_batch[:, :,0, :]
        else:
            grasp_xyz = grasp_batch[:, :, :2, :]

        # add back the offset
        grasp_xyz = grasp_xyz *scale + center

        # un-normalize
        unnormed_grasp_xyz = (
                    self.grasp_xyz_normalizer.unnormalize(grasp_xyz)
                )
        
        ##### rotation processing 
        if self.has_eff == False:
            rot6d_batch = grasp_batch[:, :, 1: , :].reshape(-1, 1, 1, 6)
        else:
            # rot6d_batch = grasp_batch[:, :, 2:, :].reshape(-1, 1, 2, 6)
            grasp_dir1 = grasp_batch[:, :, 2:4, :]
            grasp_dir2 = grasp_batch[:, :, 4:6, :]
            rot6d_batch = torch.cat((grasp_dir1, grasp_dir2), dim=-1)
            assert rot6d_batch.shape[-1] == 6

        trans_batch = self._convert_vec_to_trans(rot6d_batch, unnormed_grasp_xyz)

        trans_batch = trans_batch.detach().cpu().numpy()

        return trans_batch, unnormed_grasp_xyz, rot6d_batch

    def recover_jpose(self, jpose_batch):
        # reduce the horizon
        jpose_action = torch.mean(jpose_batch, dim=1)
        jpose_action = jpose_action.reshape(-1, self.num_eef, self.dof)
    

        unnormed_joint = (
                    self.jpose_normalizer.unnormalize(jpose_action)
                    .detach()
                    .cpu()
                    .numpy()
                )
        
        return unnormed_joint

    def forward(self, obs, history_bid = -1):
        pc = obs["pc"]
        # for real-robot, the pc is in world frame
        pc_raw_mean = pc.mean(dim=2, keepdim=True).reshape(-1, 3).detach().cpu().numpy()
        pc = pc.repeat(1, self.obs_horizon, 1, 1)
        pc = self.pc_normalizer.normalize(pc)
        
        batch_size =  pc.shape[0]

        ema_nets = self.ema.averaged_model
        feat_dict = ema_nets["encoder"](pc, ret_perpoint_feat=True, target_norm=self.pc_scale)
        center = (
            feat_dict["center"].reshape(batch_size, self.obs_horizon, 1, 3)[:, [-1]].repeat(1, self.obs_horizon, 1, 1)
        )
        # center = center[:, 0, :, :]
        scale = (
            feat_dict["scale"].reshape(batch_size, self.obs_horizon, 1, 1)[:, [-1]].repeat(1, self.obs_horizon, 1, 1)
        )
        # scale = scale[:, 0, :, :]

        pc_feat = feat_dict["so3"]  
        obs_cond_vec = pc_feat.reshape(batch_size, -1, 3)



        initial_noise_scale = 1
        
        if self.mask_type == "only_grasp": 
            # only grasp
            noisy_action = (
                torch.randn((batch_size, self.pred_horizon, self.eef_dim, 3)).to(self.device)
                * initial_noise_scale,
                None,
            )
        else: # qpose + grasp
            noisy_action = (
                torch.randn((batch_size, self.pred_horizon, self.eef_dim, 3)).to(self.device)
                * initial_noise_scale,
                torch.randn((batch_size, self.pred_horizon, self.num_eef*self.dof)).to(self.device)
                * initial_noise_scale,
            )

        curr_action = noisy_action
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        denoise_history = []
        for k in self.noise_scheduler.timesteps:
            # load from existing data statistics
            # predict noise
            noise_pred = ema_nets["noise_pred_net"](
                sample=curr_action[0],
                timestep=k,
                scalar_sample=curr_action[1],
                cond=obs_cond_vec,
            )
            ####### inverse diffusion step
            new_action = [None, None]
            new_action[0] = self.noise_scheduler.step(
                model_output=noise_pred[0], timestep=k, sample=curr_action[0]
            ).prev_sample

            if noise_pred[1] is not None:
                new_action[1] = self.noise_scheduler.step(
                    model_output=noise_pred[1], timestep=k, sample=curr_action[1]
                ).prev_sample
                   
            # record history
            if history_bid >=0:
                trans_batch, _, _ = self.recover_grasp(new_action[0], scale, center)
                assert trans_batch.shape[3] == 4
                trans_mat = trans_batch[history_bid]
                trans_mat = np.mean(trans_mat, axis=0)
                if noise_pred[1] is not None:
                    unnormed_joint = self.recover_jpose(new_action[1])
                    jpose_flat = unnormed_joint[history_bid].reshape(-1)
                    jpose_12d = np.zeros(12)
                    if self.symb_mask[0] == 'None':  # only right
                        jpose_12d[6:] = jpose_flat
                    elif self.symb_mask[1] == 'None':
                        jpose_12d[:6] = jpose_flat
                    else:
                        jpose_12d = jpose_flat
                    action_slice = (trans_mat, jpose_12d)
                else:
                    action_slice = (trans_mat, None)
                denoise_history.append(action_slice)

            # record the denoised action
            curr_action = tuple(new_action)

        trans_batch, unnormed_grasp_xyz, rot6d_batch = self.recover_grasp(new_action[0], scale, center)

        metrics = {}
        if obs['gt_grasp'] is not None:
            # calculate mes of xyz and rotation

            gt_grasp_xyz, gt_dir1, gt_dir2 = self._convert_trans_to_vec(obs['gt_grasp'])
            gt_grasp_rot6d = torch.cat((gt_dir1, gt_dir2), dim=-1)
            gt_grasp_xyz = torch.mean(gt_grasp_xyz, dim=1)
            gt_grasp_rot6d = torch.mean(gt_grasp_rot6d, dim=1)

            xyz_mse = torch.nn.functional.mse_loss(unnormed_grasp_xyz, gt_grasp_xyz)
            rot_mse = torch.nn.functional.mse_loss(rot6d_batch.reshape(gt_grasp_rot6d.shape), gt_grasp_rot6d)

            metrics = {'grasp_xyz_error': xyz_mse, 
                    'grasp_rotation_error': rot_mse}

        
        if new_action[1] is not None:
            unnormed_joint = self.recover_jpose(new_action[1])

            if obs['joint_pose'] is not None:
                # calculate joint error
                gt_joint = obs['joint_pose']
                unnormed_joint = torch.tensor(unnormed_joint, device=self.device)
                joint_mse = torch.nn.functional.mse_loss(unnormed_joint, gt_joint)
                metrics['joint_error'] = joint_mse
        
        if history_bid >= 0:
            # print the grasp xyz
            print(f"Grasp xyz: {unnormed_grasp_xyz[history_bid, 0]}")

        if len(metrics.values()) > 0:
            return  denoise_history, metrics
        
        else:
            assert batch_size == 1

            # # transform grasp pose from world frame to object frame
            # # center = torch.mean(center, dim=1).reshape(-1, 3).detach().cpu().numpy()
            # trans_batch[:, :, :3, 3] = trans_batch[:, :, :3, 3] - pc_raw_mean
            # trans_batch[:, :, 4:7, 3] = trans_batch[:, :, 4:7, 3] - pc_raw_mean

            # output final grasps and jposes
            action_dict = {}
            action_dict['grasp'] = trans_batch.reshape(-1, 4)
            if new_action[1] is not None:
                action_dict['jpose'] = unnormed_joint.reshape(self.num_eef, self.dof)

            return denoise_history, action_dict
    
    def conclude_masks(self):
        has_grasp = False
        has_jpose = False
        if self.symb_mask[0] != 'None' or self.symb_mask[1] != 'None':
            has_jpose = True
        if self.symb_mask[2] != 'None' or self.symb_mask[3] != 'None':
            has_grasp = True
        if has_grasp and has_jpose:
            return "both"
        elif has_grasp:
            return "only_grasp"
        elif has_jpose:
            return "only_jpose"