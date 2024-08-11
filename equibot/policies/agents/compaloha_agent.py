import numpy as np
import torch
from torch import nn

from equibot.policies.utils.norm import Normalizer
from equibot.policies.utils.misc import to_torch
from equibot.policies.utils.diffusion.lr_scheduler import get_scheduler

from aloha_agent import ALOHAAgent
from equibot.policies.agents.compaloha_policy import CompALOHAPolicy

class CompALOHAAgent(ALOHAAgent):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def _init_actor(self):
        self.actor = CompALOHAPolicy(self.cfg, device=self.cfg.device).to(self.cfg.device)
        self.actor.ema.averaged_model.to(self.cfg.device)

    def update(self, batch, vis=False):
        self.train()

        batch = to_torch(batch, self.cfg.device)
        pc = batch["pc"]
        joint_data  = batch["joint_pose"]
        grasp_pose = batch["grasp_pose"]

        pc = pc.repeat(1, self.obs_horizon, 1, 1)
        joint_data = joint_data.repeat(1, self.pred_horizon, 1, 1)
        left_joint_data = joint_data[:, :, 0,  :]
        right_joint_data = joint_data[:, :, 1,  :]
        grasp_pose = grasp_pose.repeat(1, self.pred_horizon, 1, 1)

        if self.pc_scale is None:
            self._init_normalizers(batch)

        # proc pc
        pc = self.pc_normalizer.normalize(pc)
        batch_size = pc.shape[0]
        feat_dict = self.actor.encoder_handle(pc, target_norm=self.pc_scale)
        center = (
            feat_dict["center"].reshape(batch_size, self.obs_horizon, 1, 3)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        )
        scale = feat_dict["scale"].reshape(batch_size, self.obs_horizon, 1, 1)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        
        # proc grasp pose. first split flattened pose to xyz and dir, then normalize xyz. 
        grasp_xyz, grasp_dir1, grasp_dir2 = self.actor._convert_grasp_to_vec(grasp_pose)
        grasp_xyz = self.grasp_xyz_normalizer.normalize(grasp_xyz)
        grasp_xyz = (grasp_xyz - center)/scale
        gt_grasp_z = torch.cat([grasp_xyz, grasp_dir1, grasp_dir2], dim=-2)

        