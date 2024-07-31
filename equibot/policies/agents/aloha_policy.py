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
        num_scalar_dims = self.dof * self.num_eef # joint pose
        # assume the symb mask is 2*[jpose, grasp, other_obj], then the activated_obsv is obj_grasp+ other_obj
        # for now, jpose require no obs
        activated_obsv = 0
        self.obs_dim = self.encoder_out_dim + activated_obsv
        self.noise_pred_net = VecConditionalUnet1D(
            input_dim=num_scalar_dims,
            cond_dim=self.obs_dim,
            scalar_cond_dim=0,
            scalar_input_dim=num_scalar_dims,
            diffusion_step_embed_dim=self.obs_dim,
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
        jpose = jpose.reshape(jpose.shape[0], jpose.shape[1],  -1, 3)
        return jpose
    
    def forward(self, obs, jpose, timesteps, symb_mask=None):
        raise NotImplementedError("Not implemented yet")
