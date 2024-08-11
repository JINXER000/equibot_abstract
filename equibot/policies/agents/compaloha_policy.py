import copy
import hydra
import torch
from torch import nn
import torch.nn.functional as F

from equibot.policies.vision.sim3_encoder import SIM3Vec4Latent
from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.equivariant_diffusion.conditional_unet1d import VecConditionalUnet1D
import numpy as np

from aloha_policy import ALOHAPolicy

class CompALOHAPolicy(ALOHAPolicy):
    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)

        # overide parameters
        num_scalar_dims = self.dof
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

    def _convert_jpose_to_vec(self, jpose, batch=None):
        # input: (B, 1, E , dof); output: (B, 1, ac_dim, 3) 
        jpose = jpose.reshape(jpose.shape[0], 1,  self.dof)
        return jpose