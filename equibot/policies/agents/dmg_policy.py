import copy
import hydra
import torch
from torch import nn
import numpy as np

from equibot.policies.vision.sim3_encoder import SIM3Vec4Latent
from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.equivariant_diffusion.conditional_unet1d import VecConditionalUnet1D
from equibot.policies.utils.equivariant_diffusion.unconditional_mlp import UnconditionalMLP
from equibot.policies.utils.lan_utils import get_and_save_skill_bert_embs, MLPEncoder

from equibot.policies.utils.misc import to_torch, \
    convert_trans_to_vec, convert_vec_to_trans, ActionSlice,\
    rotation_6d_to_matrix, geodestDist, EQUIBOT_PATH, to_torch, to_tensor,\
    convert_trans_to_4pts, convert_4pts_to_trans, matrix_to_rotation_6d, render_trajectory

import os
    
class DMGPolicy(nn.Module):
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



        if hasattr(cfg.model, "num_diffusion_iters"):
            self.num_diffusion_iters = cfg.model.num_diffusion_iters
        else:
            self.num_diffusion_iters = cfg.model.noise_scheduler.num_train_timesteps


        self.encoder_out_dim = cfg.model.encoder.c_dim

        self.separate_encoder = cfg.model.separate_encoder
        self.separate_policy = cfg.model.separate_policy

        self.dof = cfg.env.dof # 6
        self.num_eef = cfg.env.num_eef

        self.obs_dim = self.encoder_out_dim

        net_dict = {}
        if self.separate_encoder:
            self.objects = set(cfg.data.dataset.conditioned_objects)
            for obj in self.objects:
                encoder_key = f'{obj}_encoder'
                net_dict[encoder_key] = SIM3Vec4Latent(**cfg.model.encoder)
        else:
            net_dict['obj_encoder'] = SIM3Vec4Latent(**cfg.model.encoder)

        self.eef_dims = {}
        # TODO: load the skill names from statistics processed from hdf5
        self.skill_names = cfg.data.dataset.skill_names

        # self.skill_obj_mapping = {self.skill_names[i]: cfg.data.dataset.conditioned_objects[i] for i in range(len(self.skill_names))}

        if 'language_encoder_cfg' in cfg.model:
            language_encoder_cfg = cfg.model.language_encoder_cfg
            output_size = language_encoder_cfg.hidden_size
            assert output_size == self.encoder_out_dim
            self.language_encoder = self._setup_language_encoder(output_size=output_size, **language_encoder_cfg)
            net_dict['language_encoder'] = self.language_encoder 

            skill_name_to_emb = get_and_save_skill_bert_embs(self.skill_names, cache_dir=os.path.join(EQUIBOT_PATH, cfg.data.dataset.embedding_cache_dir))
            
            # Convert skill embeddings to tensors once during initialization
            self.skill_name_to_emb_tensor = to_torch(to_tensor(skill_name_to_emb), self.device)

        else:
            self.skill_scalar_mapping = {self.skill_names[i]: torch.tensor(i*100).to(self.device) for i in range(len(self.skill_names))}
            self.language_encoder = None

        ## eef_representation can be vectors or points
        self.eef_representation = cfg.data.dataset.eef_representation
        if self.eef_representation == "3vec":
            self.eef_proc_fn = self.proc_eef_3vec
            self.eef_recover_fn = self.recover_eef_3vec
            self.eef_dims = {skill_name:3 for skill_name in self.skill_names}
        elif self.eef_representation == "4pts":
            self.eef_proc_fn = self.proc_eef_4pts
            self.eef_recover_fn = self.recover_eef_4pts
            ## expand to B, H, 4, 3
            self.original_gripper_pcd = np.array(cfg.data.dataset.original_gripper_pcd)
            # self.original_gripper_pcd = np.array([[0.10, 0.01, 0], 
            #                                       [0.03, -0.04, 0],
            #                                       [0.02, 0.05, 0],
            #                                       [0,0,0]])
            
            self.eef_dims = {skill_name: 4 for skill_name in self.skill_names}
        else:
            raise ValueError(f"Unsupported eef_representation: {self.eef_representation}")

        for skill_name in self.skill_names:
            if 'bimanual' in skill_name:
                joint_scalar_dims = self.dof * self.num_eef  
                net_dict[f'{skill_name}_noise_pred_net'] = UnconditionalMLP(
                    input_dim= joint_scalar_dims,
                    diffusion_step_embed_dim=self.obs_dim* self.obs_horizon,
                )   
            else:
                ## TODO: check if we should use word embedding
                if 'unitraj_noise_pred_net' in net_dict:
                    continue

                if self.separate_policy:
                    policy_key = f'{skill_name}_noise_pred_net'
                    scalar_cond_dim = 0
                else:
                    policy_key = 'unitraj_noise_pred_net'
                    if self.language_encoder is None:
                        scalar_cond_dim= self.obs_horizon
                    else:
                        scalar_cond_dim = self.encoder_out_dim * self.obs_horizon ## TODO: check size in policy network

                net_dict[policy_key] = VecConditionalUnet1D(
                    input_dim=self.eef_dims[skill_name],  ## vec dim, rot is 2, xyz is 1
                    cond_dim=self.obs_dim* self.obs_horizon,
                    scalar_cond_dim= scalar_cond_dim,  ## if =1,  it is the skill_scalar_id
                    scalar_input_dim= 1,  ## output gripper val
                    diffusion_step_embed_dim=self.obs_dim* self.obs_horizon,
                    cond_predict_scale=False,
                    down_dims=cfg.model.down_dims,
                    )
        
        self.nets = nn.ModuleDict(net_dict)

        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        self._init_torch_compile()

        self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler)


        
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized DMG Policy with {num_parameters} parameters")

    def _init_torch_compile(self):
        nets_handles = {}
        if self.use_torch_compile:
            for key, net in self.nets.items():
                nets_handles[key] = torch.compile(net)
        else:
            nets_handles = self.nets

        self.nets = nets_handles



    def _setup_language_encoder(self, network_name, **language_encoder_kwargs):
        return eval(network_name)(**language_encoder_kwargs)
    

    def get_skill_name_encoding(self, skill_name, batch_size):
        if self.language_encoder is None:
            skill_scalar_id = self.skill_scalar_mapping[skill_name].repeat(batch_size,1)
        else:
            skill_emb_tensor = self.skill_name_to_emb_tensor[skill_name]
            skill_emb = self.nets['language_encoder'](skill_emb_tensor)
            skill_scalar_id = skill_emb.expand(batch_size, -1)
        return skill_scalar_id

    def step_ema(self):
        self.ema.step(self.nets)

    def normalize_from_key(self, key, data):
        return self.all_normalizers[key].normalize(data)
    
    def unnormalize_from_key(self, key, data):
        return self.all_normalizers[key].unnormalize(data)

    def recover_eef_3vec(self, eefpos_batch, scale, center, key):
        side = key.split('_')[0]

        # reshape dim to B,  (3 or 6) , 3
        # eefpos_batch = torch.mean(eefpos_batch, dim=1, keepdim=True)
        scale = torch.mean(scale, dim=2, keepdim=True)
        center = torch.mean(center, dim=2, keepdim=True)

        batch_size = eefpos_batch.shape[0]

        ##### eefpos processing
        eefpos_xyz = eefpos_batch[:, :,0, :].reshape(batch_size, -1, 1, 3)

        # # add back the offset
        eefpos_xyz = eefpos_xyz *scale + center

        # un-normalize
        unnormed_eefpos_xyz = (
                    self.unnormalize_from_key(key, eefpos_xyz)
                )
        
        ##### rotation processing 
        rot6d_batch = eefpos_batch[:, :, 1: , :].reshape(batch_size, -1, 1, 6)

        trans_batch = convert_vec_to_trans(rot6d_batch, unnormed_eefpos_xyz)

        # trans_batch = trans_batch.detach().cpu().numpy()

        return trans_batch, unnormed_eefpos_xyz, rot6d_batch
    
    ## eefpt_batch is B, H, 4, 3
    def recover_eef_4pts(self, eefpt_batch, scale, center, key):
        scale = torch.mean(scale, dim=2, keepdim=True)
        center = torch.mean(center, dim=2, keepdim=True)
        
        eefpt_batch = eefpt_batch * scale + center
        unnormed_eefpt_batch = self.unnormalize_from_key(key, eefpt_batch)

        trans_batch = convert_4pts_to_trans(unnormed_eefpt_batch, self.original_gripper_pcd)
        
        ## only for eval
        unnormed_eefpos_xyz = trans_batch[:, :, :3, 3:].transpose(-2, -1)
        eef_rot = trans_batch[:, :, :3, :3].reshape(-1, 3, 3)
        rot6d_batch = matrix_to_rotation_6d(eef_rot)

        # trans_batch = trans_batch.detach().cpu().numpy()
        return trans_batch, unnormed_eefpos_xyz, rot6d_batch

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

    def proc_pc(self, pc, skill_name, ema_nets = None):
        pc_key = f'{skill_name}:pc'
        pc = self.normalize_from_key(pc_key, pc)
        batch_size = pc.shape[0]

        ## in training
        if self.separate_encoder:
            obj_name = self.skill_obj_mapping[skill_name]
            encoder_key = f'{obj_name}_encoder'
        else:
            encoder_key = 'obj_encoder'

        encoder_handle = self.nets[encoder_key] 
        if ema_nets is None:
            feat_dict = encoder_handle(pc, target_norm=self.all_normalizers[f'{skill_name}:pc_scale'])
        else: # in inference
            feat_dict = encoder_handle(pc, ret_perpoint_feat=False, target_norm=self.all_normalizers[f'{skill_name}:pc_scale'])
        
        center = (
            feat_dict["center"].reshape(batch_size, self.obs_horizon, 1, 3)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        )
        scale = feat_dict["scale"].reshape(batch_size, self.obs_horizon, 1, 1)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        equiv_feat = feat_dict["so3"]  
        obs_cond_vec = equiv_feat.reshape(batch_size, -1, 3)
        return obs_cond_vec, center, scale

    # in dataset, first pc is converted using min(). Then, in pc_normalizer, pc.max is mapped to 1. in eef normalizer, eef_xyz = traj = (traj-pc.min)/pc.max.  Here center should be 0.5, and scale be 1. finally, eef_xyz mean shoule be near 0. 
    def proc_eef_3vec(self, eef_pose, key, center, scale):
        eef_xyz_raw, eef_dir1, eef_dir2 = convert_trans_to_vec(eef_pose)
        eef_xyz = self.normalize_from_key(key, eef_xyz_raw)
        eef_xyz = (eef_xyz - center)/scale
        gt_eef_z = torch.cat([eef_xyz, eef_dir1, eef_dir2], dim=-2)

        return gt_eef_z
    
    def proc_eef_4pts(self, eef_pose, key, center, scale):
        eef_4pts_raw = convert_trans_to_4pts(eef_pose, self.original_gripper_pcd)
        eef_4pts = self.normalize_from_key(key, eef_4pts_raw)
        eef_4pts = (eef_4pts - center) / scale

        # ## debug to check if the eef_4pts is correct
        # trans_batch_recon, _, _ = self.recover_eef_4pts(eef_4pts, scale, center, key)
        # trans_batch_recon_tensor = torch.tensor(trans_batch_recon).to(self.device)
        # import torch.nn.functional as F
        # error = F.mse_loss(trans_batch_recon_tensor, eef_pose)
        # print(f"Error: {error}")
        # assert torch.allclose(trans_batch_recon_tensor, eef_pose, atol=1e-4)
        return eef_4pts

    def proc_gripper(self, raw_gripper, key):
        gripper_action = self.normalize_from_key(key, raw_gripper)
        gripper_action = gripper_action.reshape(gripper_action.shape[0], -1, 1)
        return gripper_action
    
    def proc_jpose(self, jpose, key):
        
        jpose_n = self.normalize_from_key(key, jpose)
        # jpose_vec = self._convert_jpose_to_vec(jpose_n)
        jpose_vec = jpose_n.reshape(jpose_n.shape[0], -1,  self.dof * self.num_eef)
        return jpose_vec
    
    def pred_bimanual_jposes(self, skill_name, batch_size, gt_batch = None):

        ema_nets = self.ema.averaged_model

        initial_noise_scale = 1
        noisy_jpose = torch.randn((batch_size,   self.num_eef*self.dof)).to(self.device) * initial_noise_scale

        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        curr_action = {f'{skill_name}:jpose': noisy_jpose}

        ####### inverse diffusion step
        for k in self.noise_scheduler.timesteps:
            biop_key = f'{skill_name}:jpose'
            new_action = {biop_key: None}

            scalar_noise_pred = ema_nets[f'{skill_name}_noise_pred_net'](\
                sample=curr_action[f'{skill_name}:jpose'],
                timesteps = k,
            )
            new_action[biop_key] = self.noise_scheduler.step(
                model_output=scalar_noise_pred, timestep=k, sample=curr_action[biop_key]
            ).prev_sample

            curr_action = new_action

        unnormed_joint = self.recover_jpose(curr_action[biop_key], key=biop_key)
        unnormed_joint = torch.tensor(unnormed_joint).to(self.device)   

        action_dict = {}
        eval_metrics = {}
        if batch_size ==1:
            action_dict[biop_key] = unnormed_joint.reshape(self.num_eef, self.dof)
        else:
            gt_joint = gt_batch[f'{skill_name}:jpose'].reshape(-1, self.num_eef, self.dof)
            joint_mse = torch.nn.functional.mse_loss(unnormed_joint, gt_joint)
            eval_metrics["dual_joint_mse"] = joint_mse

        return action_dict, eval_metrics
    
    def pred_unimaual_traj(self, skill_name, agent_obs, gt_batch = None):
        pc_data = agent_obs[f'{skill_name}:pc'].repeat(1, self.obs_horizon, 1, 1)
        batch_size =  pc_data.shape[0]

        ema_nets = self.ema.averaged_model

        obs_vec, center, scale = self.proc_pc(pc_data, skill_name, ema_nets = ema_nets)

        ##### start denoising #####

        initial_noise_scale = 1
        noisy_eef_xt = torch.randn((batch_size, self.pred_horizon, self.eef_dims[skill_name], 3)).to(self.device)\
        * initial_noise_scale

        noisy_gripper = torch.randn((batch_size, self.pred_horizon, 1)).to(self.device) * initial_noise_scale

        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        curr_action = { f"{skill_name}:eefpos": noisy_eef_xt, 
            f"{skill_name}:gripper": noisy_gripper}
        
         ####### inverse diffusion step
        if self.separate_policy:
            policy_key = f'{skill_name}_noise_pred_net'
            skill_scalar_id = None
        else:
            policy_key = 'unitraj_noise_pred_net'
            skill_scalar_id = self.get_skill_name_encoding(skill_name, batch_size)
            
        for k in self.noise_scheduler.timesteps:

            new_action = {f"{skill_name}:eefpos": None, f"{skill_name}:gripper": None }

            vec_noise_pred, gripper_noise_pred = ema_nets[policy_key](\
                sample=curr_action[f"{skill_name}:eefpos"],
                timestep = k,
                scalar_sample = curr_action[f"{skill_name}:gripper"], 
                cond= obs_vec,
                scalar_cond=skill_scalar_id,
            )
            new_action[f"{skill_name}:eefpos"] = self.noise_scheduler.step(
                model_output=vec_noise_pred, timestep=k, sample=curr_action[f"{skill_name}:eefpos"]
            ).prev_sample

            new_action[f"{skill_name}:gripper"] = self.noise_scheduler.step(
                model_output=gripper_noise_pred, timestep=k, sample=curr_action[f"{skill_name}:gripper"]
            ).prev_sample

            curr_action = new_action

        ### recover the eefpos eef pose
        ## predicted values
        trans_batch, unnormed_eefpos_xyz, rot6d_batch = self.eef_recover_fn(\
            curr_action[f"{skill_name}:eefpos"], scale, center, key=f"{skill_name}:eefpos")
        assert trans_batch.shape[3] == 4

        ## recover gripper
        gripper_batch = self.recover_gripper(curr_action[f"{skill_name}:gripper"], key=f"{skill_name}:gripper")

        ## update action dict 
        action_dict = {}
        eval_metrics = {}
        if batch_size ==1:
            action_dict[f"{skill_name}:eefpos"] = trans_batch[0].detach().cpu().numpy()
            action_dict[f"{skill_name}:gripper"] = gripper_batch[0]
        ## calc metrics if in training
        else:
            if self.eef_representation == "4pts":
                gt_4pts =  self.eef_proc_fn(gt_batch[f"{skill_name}:eefpos"], f'{skill_name}:eefpos', center, scale)
                pred_4pts = curr_action[f"{skill_name}:eefpos"]
                pts_error = torch.nn.functional.mse_loss(pred_4pts, gt_4pts)
                eval_metrics[f"{skill_name}:pts_error"] = pts_error

            pred_xyz = trans_batch[:, :, :3, 3]
            gt_xyz = gt_batch[f"{skill_name}:eefpos"][:, :, :3, 3]
            xyz_l1 = torch.nn.functional.l1_loss(pred_xyz, gt_xyz)
            eval_metrics[f"{skill_name}:xyz_l1"] = xyz_l1

            gt_Rs = gt_batch[f"{skill_name}:eefpos"][:, :, :3, :3]
            pred_Rs = trans_batch[:, :, :3, :3]
            diff_theta = geodestDist(gt_Rs, pred_Rs).mean()
            eval_metrics[f"{skill_name}:rot_diff"] = diff_theta * 180 / torch.pi

            ## render the eefpos and pc
            # Get the point cloud data and trajectory
            pc_data = agent_obs[f'{skill_name}:pc'][0,0].detach().cpu().numpy()  # Shape: (N, 3)
            trajectory = trans_batch[0].detach().cpu().numpy()  # Shape: (T, 4, 4)
            gripper_values = gripper_batch[0]  # Shape: (T,)
            # Render trajectory and PC together
            rendered_img = render_trajectory(pc_data, trajectory,  gripper_values, title = f'{skill_name}-prediction')

            ## render the gt trajectory also
            gt_trajectory = gt_batch[f"{skill_name}:eefpos"][0].detach().cpu().numpy()
            gt_gripper = gt_batch[f"{skill_name}:gripper"][0].detach().cpu().numpy()
            gt_rendered_img = render_trajectory(pc_data, gt_trajectory, gt_gripper, title = f'{skill_name}-ground_truth')
            
            # Store the rendered image in eval_metrics
            eval_metrics[f"{skill_name}:image"] = rendered_img

        return action_dict, eval_metrics




    def forward(self, batch, skill_id=-1):
        ###### preprocess data from dataset #######
        batch = to_torch(batch, self.device)

        action_dict_all = {}
        eval_metrics_all = {}
        for skill_name in self.skill_names:
            
            if 'bimanual' in skill_name:
                batch_size = batch[f'{skill_name}:jpose'].shape[0]
                action_dict, eval_metrics = self.pred_bimanual_jposes(skill_name, batch_size=batch_size, gt_batch=batch)
            else:
                pc_data = batch[f'{skill_name}:pc']
                agent_obs = {f'{skill_name}:pc': pc_data}
                action_dict, eval_metrics = self.pred_unimaual_traj(skill_name, agent_obs, gt_batch=batch)
            action_dict_all.update(action_dict)
            eval_metrics_all.update(eval_metrics)

        denoise_history = []
        if skill_id >= 0 and len(action_dict_all) > 0:
            ## in traj mode, we do not visulize the history. Instead, we visualize the final action
            skill_name = self.skill_names[skill_id]
            traj_len = self.pred_horizon
            for i in range(traj_len):
                action_slice = ActionSlice(mode="separated")
                if 'bimanual' in skill_name:
                    action_slice.update(f"{skill_name}:jpose", action_dict_all[f'{skill_name}:jpose'].reshape(-1))
                else:
                    action_slice.update(f"{skill_name}:eefpos", action_dict_all[f"{skill_name}:eefpos"][i])
                    action_slice.update(f"{skill_name}:gripper", action_dict_all[f"{skill_name}:gripper"][i])

                denoise_history.append(action_slice)
        return action_dict_all, eval_metrics_all, denoise_history

 