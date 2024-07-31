import numpy as np
import torch
from torch import nn

from equibot.policies.utils.norm import Normalizer
from equibot.policies.utils.misc import to_torch
from equibot.policies.utils.diffusion.lr_scheduler import get_scheduler

from equibot.policies.agents.aloha_policy import ALOHAPolicy

class ALOHAAgent(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._init_actor()
        if cfg.mode == "train":
            self.optimizer = torch.optim.AdamW(
                self.actor.nets.parameters(),
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
            )
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_warmup_steps=500,
                num_training_steps=cfg.data.dataset.num_training_steps,
            )
        self.device = cfg.device
        self.num_eef = cfg.env.num_eef
        self.dof = cfg.env.dof
        self.num_points = cfg.data.dataset.num_points
        self.obs_mode = cfg.model.obs_mode
        self.ac_mode = cfg.model.ac_mode
        self.obs_horizon = cfg.model.obs_horizon
        self.pred_horizon = cfg.model.pred_horizon
        self.ac_horizon = cfg.model.ac_horizon
        self.shuffle_pc = cfg.data.dataset.shuffle_pc

        self.pc_normalizer = None
        self.xyz_normalizer = None
        self.jpose_normalizer = None

        self.pc_scale = None

        self.symb_mask = cfg.data.dataset.symb_mask


    def _init_actor(self):
        self.actor = ALOHAPolicy(self.cfg, device=self.cfg.device).to(self.cfg.device)
        self.actor.ema.averaged_model.to(self.cfg.device)

     # TODO: adapt to aloha
    def _init_normalizers(self, batch):
        if self.jpose_normalizer is None: # normalize to [0, max]
            joint_data  = batch["joint_pose"]
            flattend_joint_data = joint_data.view(-1, self.dof)
            indices = [[0, 1, 2, 3, 4, 5]]


            jpose_normalizer = Normalizer(
                flattend_joint_data, symmetric=True, indices=indices
            )
            self.jpose_normalizer = Normalizer(
                {
                    "min": jpose_normalizer.stats["min"],
                    "max": jpose_normalizer.stats["max"],
                }
            )
            print(f"Joint pose normalization stats: {self.jpose_normalizer.stats}")

        if self.xyz_normalizer is None:
            self.xyz_normalizer = Normalizer(
            {
                "min": jpose_normalizer.stats["min"][:3],
                "max": jpose_normalizer.stats["max"][:3],
            }

            )
        if self.pc_normalizer is None:
            self.pc_normalizer = self.xyz_normalizer
            self.actor.pc_normalizer = self.pc_normalizer

        # compute action scale relative to point cloud scale
        pc = batch["pc"].reshape(-1, self.num_points, 3)
        centroid = pc.mean(1, keepdim=True)
        centered_pc = pc - centroid
        pc_scale = centered_pc.norm(dim=-1).mean()
        ac_scale = jpose_normalizer.stats["max"].max()
        self.pc_scale = pc_scale / ac_scale
        self.actor.pc_scale = self.pc_scale

    def train(self, training=True):
        self.actor.nets.train(training)

    def update(self, batch, vis=False):
        self.train()

        batch = to_torch(batch, self.device)
        pc = batch["pc"]
        joint_data  = batch["joint_pose"]

        if self.pc_scale is None:
            self._init_normalizers(batch)
        pc = self.pc_normalizer.normalize(pc)
        joint_data = self.jpose_normalizer.normalize(joint_data)

        batch_size = pc.shape[0]

        feat_dict = self.actor.encoder_handle(pc, target_norm=self.pc_scale)
        center = (
            feat_dict["center"].reshape(batch_size, 1, 3)[:, [-1]].repeat(1, 1, 1)
        )
        scale = feat_dict["scale"].reshape(batch_size, 1, 1)[:, [-1]].repeat(1, 1, 1)
        
        # scalar
        scalar_jpose = self.actor._convert_jpose_to_vec(joint_data)
        
        obs_cond_vec = feat_dict["so3"]  
        obs_cond_vec = obs_cond_vec.reshape(batch_size,  -1, 3)

        timesteps = torch.randint(
            0,
            self.actor.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()

        scalar_jpose_noise = torch.randn_like(scalar_jpose, device=self.device)
        noisy_jpose = self.actor.noise_scheduler.add_noise(
            scalar_jpose, scalar_jpose_noise, timesteps
        )
        # NOTE: noisy_jpose should be  scalar_sample, but here it is set as sample as it is compulsory
        scalar_jpose_pred = self.actor.noise_pred_net_handle(
            noisy_jpose,
            timesteps,
            scalar_sample = None, 
            cond=obs_cond_vec,
            scalar_cond=None,
        )
        if self.symb_mask[2] == 'None' and self.symb_mask[3] == 'None':
            loss = nn.functional.mse_loss(scalar_jpose_pred, scalar_jpose_noise)
        else:
            raise NotImplementedError("Not implemented yet")
        if torch.isnan(loss):
            print(f"Loss is nan, please investigate.")
            import pdb

            pdb.set_trace()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.actor.step_ema()

        metrics = {
            "loss": loss,
            "mean_gt_noise_norm": np.linalg.norm(
                scalar_jpose_noise.reshape(joint_data.shape[0], -1).detach().cpu().numpy(),
                axis=1,
            ).mean(),
            "mean_pred_noise_norm": np.linalg.norm(
                scalar_jpose_pred.reshape(joint_data.shape[0], -1)
                .detach()
                .cpu()
                .numpy(),
                axis=1,
            ).mean(),
        }

        return metrics