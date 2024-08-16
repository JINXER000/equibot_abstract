import copy
import hydra
import torch
from torch import nn
import torch.nn.functional as F

from equibot.policies.vision.sim3_encoder import SIM3Vec4Latent
from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.equivariant_diffusion.conditional_unet1d import VecConditionalUnet1D
import numpy as np

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

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

        if hasattr(cfg.model, "num_diffusion_iters"):
            self.num_diffusion_iters = cfg.model.num_diffusion_iters
        else:
            self.num_diffusion_iters = cfg.model.noise_scheduler.num_train_timesteps

        if self.obs_mode.startswith("pc"):
            self.encoder = SIM3Vec4Latent(**cfg.model.encoder) # hidden_dim = 32
        else:
            self.encoder = None
        self.encoder_out_dim = cfg.model.encoder.c_dim

        self.num_eef = cfg.env.num_eef
        self.dof = cfg.env.dof # 6
        self.eef_dim = 3 # xyz, dir1, dir2
        num_scalar_dims = self.dof * self.num_eef # joint pose
        # assume the symb mask is 2*[jpose, grasp, other_obj], then the activated_obsv is obj_grasp+ other_obj
        # for now, jpose require no obs
        activated_obsv = 0
        self.obs_dim = self.encoder_out_dim + activated_obsv
        self.noise_pred_net = VecConditionalUnet1D(
            input_dim=self.eef_dim,
            cond_dim=self.obs_dim* self.obs_horizon,
            scalar_cond_dim=0,
            scalar_input_dim= 0 if self.symb_mask[0] == 'None' and self.symb_mask[1] == 'None' else num_scalar_dims,
            diffusion_step_embed_dim=self.obs_dim* self.obs_horizon,
            cond_predict_scale=True,
        )

        self.nets = nn.ModuleDict(
            {"encoder": self.encoder, "noise_pred_net": self.noise_pred_net}
        )
        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        self._init_torch_compile()

        self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler)

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
    
    def _convert_grasp_to_vec(self, grasp, batch = None):
        assert grasp.shape[-1] == 9
        eef_pose = grasp[:, :, :, :3]
        dir1 = grasp[:, :, :, 3:6]
        dir2 = grasp[:, :, :, 6:9]

        return eef_pose, dir1, dir2
    
    def _convert_vec_to_grasp(self, g_vec):
        eef_pose = g_vec[:, :, 0, :]
        dir1 = g_vec[:, :, 1, :]
        dir2 = g_vec[:, :, 2, :]

        grasp_flatten = torch.cat([eef_pose, dir1, dir2], dim = -1)

        return grasp_flatten
    
    def step_ema(self):
        self.ema.step(self.nets)


    def recover_grasp(self, grasp_batch, scale, center, history_bid = -1):
        # reduce dim to B, 3, 3
        grasp_action = torch.mean(grasp_batch, dim=1)
        # add back the offset
        grasp_xyz = grasp_action[:,0].reshape(-1, 1, 3)
        grasp_xyz = grasp_xyz *scale + center

        # un-normalize
        unnormed_grasp_xyz = (
                    self.grasp_xyz_normalizer.unnormalize(grasp_xyz)
                    .detach()
                    .cpu()
                    .numpy()
                )
        rot6d_batch = grasp_action[:, 1:].reshape(-1, 1, 6)
        # pt3d version of f_gs()
        rot6d = rot6d_batch[history_bid]
        rotation_mat_ts = rotation_6d_to_matrix(rot6d)
        rotation_mat = rotation_mat_ts.cpu().numpy()

        trans_mat = np.eye(4)
        trans_mat[:3, :3] = rotation_mat
        trans_mat[:3, 3] = unnormed_grasp_xyz[history_bid]

        return trans_mat, unnormed_grasp_xyz, rot6d_batch

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
        pc = pc.repeat(1, self.obs_horizon, 1, 1)
        pc = self.pc_normalizer.normalize(pc)
        
        batch_size =  pc.shape[0]

        ema_nets = self.ema.averaged_model
        feat_dict = ema_nets["encoder"](pc, ret_perpoint_feat=True, target_norm=self.pc_scale)
        center = (
            feat_dict["center"].reshape(batch_size, self.obs_horizon, 1, 3)[:, [-1]].repeat(1, self.obs_horizon, 1, 1)
        )
        center = center[:, 0, :, :]
        scale = (
            feat_dict["scale"].reshape(batch_size, self.obs_horizon, 1, 1)[:, [-1]].repeat(1, self.obs_horizon, 1, 1)
        )
        scale = scale[:, 0, :, :]

        pc_feat = feat_dict["so3"]  
        obs_cond_vec = pc_feat.reshape(batch_size, -1, 3)



        initial_noise_scale = 1
        
        if self.symb_mask[0] == 'None' and self.symb_mask[1] == 'None':
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
                trans_mat, _, _ = self.recover_grasp(new_action[0], scale, center, history_bid)
                if noise_pred[1] is not None:
                    unnormed_joint = self.recover_jpose(noise_pred[1])
                    action_slice = (trans_mat, unnormed_joint[0])
                else:
                    action_slice = (trans_mat, None)
                denoise_history.append(action_slice)

            # record the denoised action
            curr_action = tuple(new_action)

        # calculate mes of xyz and rot
        _, unnormed_grasp_xyz, rot6d_batch = self.recover_grasp(new_action[0], scale, center)
        
        gt_grasp_xyz = obs['gt_grasp'][:, 0, 0, :3].reshape(-1, 1, 3)
        gt_grasp_rot6d = obs['gt_grasp'][:, 0, 0, 3:].reshape(-1, 1, 6)

        xyz_mse = torch.nn.functional.mse_loss(torch.tensor(unnormed_grasp_xyz), torch.tensor(gt_grasp_xyz))
        rot_mse = torch.nn.functional.mse_loss(rot6d_batch.detach().cpu(), torch.tensor(gt_grasp_rot6d))

        metrics = {'grasp_xyz_error': xyz_mse, 
                   'grasp_rotation_error': rot_mse}
        return  denoise_history, metrics