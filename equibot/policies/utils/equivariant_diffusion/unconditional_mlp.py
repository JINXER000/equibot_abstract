from typing import Union
import torch
import torch.nn as nn
import einops

from equibot.policies.utils.diffusion.positional_embedding import SinusoidalPosEmb


class UnconditionalMLP(nn.Module):
    def __init__(self, input_dim, diffusion_step_embed_dim=128, cfg=None): 
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

        # Build network architecture based on configuration
        if cfg is not None and hasattr(cfg, 'architecture'):
            self._build_architecture_from_config(input_dim, dsed, cfg)
        else:
            # self._build_wider_architecture(input_dim, dsed)
            self._build_small_architecture(input_dim, dsed)

    def _build_architecture_from_config(self, input_dim, dsed, cfg):
        """Build network architecture based on configuration"""
        if cfg.architecture == "small":
            self._build_small_architecture(input_dim, dsed)
        elif cfg.architecture == "deeper":
            self._build_deeper_architecture(input_dim, dsed)
        elif cfg.architecture == "wider":
            self._build_wider_architecture(input_dim, dsed)
        elif cfg.architecture == "custom":
            self._build_custom_architecture(input_dim, dsed, cfg)
        else:
            raise ValueError(f"Unsupported architecture: {cfg.architecture}")

    def _build_small_architecture(self, input_dim, dsed):
        """Build small architecture (commented out in original)"""
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

    def _build_deeper_architecture(self, input_dim, dsed):
        """Build deeper architecture (commented out in original)"""
        self.jpose_encoder = nn.Sequential(
            nn.Linear(input_dim, dsed//2),
            nn.SiLU(),
            nn.Linear(dsed//2, dsed//2),
            nn.SiLU(),
            nn.Linear(dsed//2, dsed//2),
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
            nn.Linear(dsed//2, dsed//2),
            nn.SiLU(),
            nn.Linear(dsed//2, dsed//2),
            nn.SiLU(),
            nn.Linear(dsed//2, input_dim),
        )

    def _build_wider_architecture(self, input_dim, dsed):
        """Build wider architecture (currently active in original)"""
        self.jpose_encoder = nn.Sequential(
            nn.Linear(input_dim, dsed),
            nn.SiLU(),
            nn.Linear(dsed, dsed),
        )
        self.jpose_time_mixer = nn.Sequential(
            nn.Linear(2*dsed, dsed),
            nn.SiLU(),
        )
        self.jpose_decoder = nn.Sequential(
            nn.Linear(dsed, dsed),
            nn.SiLU(),
            nn.Linear(dsed, input_dim),
        )

    def _build_custom_architecture(self, input_dim, dsed, cfg):
        """Build custom architecture based on layer specifications"""
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in cfg.custom_encoder_layers:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU()
            ])
            prev_dim = hidden_dim
        self.jpose_encoder = nn.Sequential(*encoder_layers)

        # Build time mixer
        time_mixer_layers = []
        prev_dim = 2 * dsed  # Concatenated sample and time embeddings
        for hidden_dim in cfg.custom_time_mixer_layers:
            time_mixer_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU()
            ])
            prev_dim = hidden_dim
        self.jpose_time_mixer = nn.Sequential(*time_mixer_layers)

        # Build decoder
        decoder_layers = []
        prev_dim = cfg.custom_time_mixer_layers[-1]  # Output from time mixer
        for hidden_dim in cfg.custom_decoder_layers:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU()
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.jpose_decoder = nn.Sequential(*decoder_layers)

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


