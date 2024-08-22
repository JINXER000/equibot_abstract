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

        ###### Load data, preprocessing ######
        batch = to_torch(batch, self.device)
        pc = batch["pc"]
        joint_data  = batch["joint_pose"]
        grasp_pose = batch["grasp_pose"]
        pc = pc.repeat(1, self.obs_horizon, 1, 1)
        joint_data = joint_data.repeat(1, self.pred_horizon, 1, 1)
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
        grasp_xyz_raw, grasp_dir1, grasp_dir2 = self.actor._convert_trans_to_vec(grasp_pose)
        
        
        grasp_xyz = self.grasp_xyz_normalizer.normalize(grasp_xyz_raw)
        grasp_xyz = (grasp_xyz - center)/scale


        gt_grasp_z = torch.cat([grasp_xyz, grasp_dir1, grasp_dir2], dim=-2)

        # scalar
        joint_data = self.jpose_normalizer.normalize(joint_data)
        scalar_jpose = self.actor._convert_jpose_to_vec(joint_data)
        # scalar_jpose = joint_data
        
        pc_feat = feat_dict["so3"]  
        obs_cond_vec = pc_feat.reshape(batch_size, -1, 3)

        timesteps = torch.randint(
            0,
            self.actor.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()

        ######## train the pred net ########

        metrics = {
                   "cond_obsv": np.linalg.norm(
                       obs_cond_vec.detach().cpu().numpy(), axis=1
                   ).mean(),
                   }
        if self.actor.mask_type != 'only_grasp': # pred joint
            scalar_jpose_noise = torch.randn_like(scalar_jpose, device=self.device)
            noisy_jpose = self.actor.noise_scheduler.add_noise(
                scalar_jpose, scalar_jpose_noise, timesteps
            )
            scalar_jpose_noise_pred = self.actor.jpose_noise_pred_net_handle(
                torch.randn_like(noisy_grasp),
                timesteps,
                scalar_sample = noisy_jpose, 
                cond=obs_cond_vec,
                scalar_cond=None,
            )
            scalar_loss = nn.functional.mse_loss(scalar_jpose_noise_pred, scalar_jpose_noise)
            metrics["scalar_loss"] = scalar_loss


        if self.actor.mask_type != 'only_jpose':  # pred grasp
            vec_grasp_noise = torch.randn_like(gt_grasp_z, device=self.device)  
            noisy_grasp = self.actor.noise_scheduler.add_noise(
                gt_grasp_z, vec_grasp_noise, timesteps
            )      
            vec_grasp_noise_pred = self.actor.grasp_noise_pred_net_handle(
                noisy_grasp,
                timesteps,
                scalar_sample = None, 
                cond=obs_cond_vec,
                scalar_cond=None,
            )
            vec_loss= nn.functional.mse_loss(vec_grasp_noise_pred, vec_grasp_noise)
            metrics["vec_loss"] = vec_loss

        if self.actor.mask_type == 'both':
            # sum up the loss
            n_vec = np.prod(vec_grasp_noise_pred.shape)  # 3*3
            n_scalar = np.prod(scalar_jpose_noise_pred.shape)  # 2*6
            k = n_vec / (n_vec + n_scalar)
            loss = k * vec_loss + (1 - k) * scalar_loss
            metrics["loss"] = loss

        if torch.isnan(loss):
            print(f"Loss is nan, please investigate.")
            import pdb

            pdb.set_trace()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.actor.step_ema()

        return metrics