from typing import Union
import torch
import torch.nn as nn
import einops

from equibot.policies.utils.diffusion.positional_embedding import SinusoidalPosEmb


class UnconditionalMLP(nn.Module):
    def __init__(self, input_dim, diffusion_step_embed_dim=128,): 
        super().__init__()

        self.is_dummy = False
        if input_dim== 0:
            self.is_dummy = True
            return
        
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.SiLU(),
            nn.Linear(dsed * 4, dsed),
        )
        self.jpose_encoder = nn.Sequential(
            nn.Linear(input_dim, dsed//2),
            nn.SiLU(),
            nn.Linear(dsed//2, dsed),
        )
        self.jpose_time_mixer = nn.Sequential(
            nn.Linear(2*dsed, dsed),
            nn.SiLU(),
        )
        self.jpose_decoder = nn.Sequential(
            nn.Linear(dsed, dsed//2),
            nn.SiLU(),
            nn.Linear(dsed//2, input_dim),
        )

    def forward(self, 
                sample: torch.Tensor,
                 timesteps: Union[torch.Tensor, float, int],
                 ):
        if self.is_dummy:
            return None

        if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
            
        time_emb = self.diffusion_step_encoder(timesteps)

        sample_emb = self.jpose_encoder(sample)
        emb = torch.cat([sample_emb, time_emb], dim=-1)

        emb_noise = self.jpose_time_mixer(emb)

        sample_noise = self.jpose_decoder(emb_noise)

        return sample_noise


