import copy
import hydra
import torch
from torch import nn
import torch.nn.functional as F

from equibot.policies.vision.sim3_encoder import SIM3Vec4Latent
from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.equivariant_diffusion.conditional_unet1d import VecConditionalUnet1D


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
        self.action_dim = 1 # TODO: now is only xyz. 
        num_scalar_dims = self.dof * self.num_eef # joint pose
        # assume the symb mask is 2*[jpose, grasp, other_obj], then the activated_obsv is obj_grasp+ other_obj
        # for now, jpose require no obs
        activated_obsv = 0
        self.obs_dim = self.encoder_out_dim + activated_obsv
        self.noise_pred_net = VecConditionalUnet1D(
            input_dim=self.action_dim,
            cond_dim=self.obs_dim* self.obs_horizon,
            scalar_cond_dim=0,
            scalar_input_dim=num_scalar_dims,
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
        jpose = jpose.reshape(jpose.shape[0], -1,  12)
        return jpose
    
    def step_ema(self):
        self.ema.step(self.nets)

    def forward(self, obs, return_history = True):
        pc = obs["pc"]
        pc = self.pc_normalizer.normalize(pc)
        pc = pc.repeat(1, self.obs_horizon, 1, 1)

        batch_size =  pc.shape[0]

        ema_nets = self.ema.averaged_model
        feat_dict = ema_nets["encoder"](pc, target_norm=self.pc_scale)
        center = (
            feat_dict["center"].reshape(batch_size, self.obs_horizon, 1, 3)[:, [-1]].repeat(1, self.obs_horizon, 1, 1)
        )
        scale = (
            feat_dict["scale"].reshape(batch_size, self.obs_horizon, 1, 1)[:, [-1]].repeat(1, self.obs_horizon, 1, 1)
        )

        pc_feat = feat_dict["so3"]  
        obs_cond_vec = pc_feat.reshape(batch_size, -1, 3)
        initial_noise_scale = 1
        state_dim = 0
        noisy_action = (
            torch.randn((batch_size, self.pred_horizon, self.action_dim, 3)).to(self.device)
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
            # inverse diffusion step
            new_action = [None, None]
            new_action[0] = self.noise_scheduler.step(
                model_output=noise_pred[0], timestep=k, sample=curr_action[0]
            ).prev_sample
            if noise_pred[1] is not None:
                new_action[1] = self.noise_scheduler.step(
                    model_output=noise_pred[1], timestep=k, sample=curr_action[1]
                ).prev_sample
            curr_action = tuple(new_action)
            
            if return_history:
                grasp_slice = new_action[0][0][0]
                joint_slice = new_action[1][0][0].reshape(1, 1, self.num_eef, self.dof)
                action_slice = (grasp_slice, joint_slice)
                denoise_history.append(action_slice)


        # grasp_action = curr_action[0].permute(0, 2, 3, 1)
        # joint_action = curr_action[1].permute(0, 2, 1)
        grasp_action, joint_action = curr_action
        joint_action = joint_action.reshape(batch_size, self.pred_horizon, self.num_eef, self.dof)
        ret = {"grasp_action": grasp_action, "joint_action": joint_action}
        return ret, denoise_history 