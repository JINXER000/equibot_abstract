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

        ###### Load data, preprocessing using mask ######
        batch = to_torch(batch, self.device)
        right_pc = batch["right_pc"]
        right_grasp = batch["right_grasp"]
        right_jpose = batch["right_jpose"]
        left_jpose = batch["left_jpose"]

        # TODO: revise this part for other dataset
        pc = right_pc
        grasp_pose = right_grasp

        pc = pc.repeat(1, self.obs_horizon, 1, 1)
        grasp_pose = grasp_pose.repeat(1, self.pred_horizon, 1, 1)
        left_jpose = left_jpose.repeat(1, self.pred_horizon, 1, 1)
        right_jpose = right_jpose.repeat(1, self.pred_horizon, 1, 1)

        ### init normalizers
        obs_dict = {'pc': pc, 'grasp_pose': grasp_pose, 'joint_data': left_jpose}
        if self.pc_scale is None:
            self._init_normalizers(obs_dict)

        ### proc pc
        pc = self.pc_normalizer.normalize(pc)
        batch_size = pc.shape[0]
        feat_dict = self.actor.encoder_handle(pc, target_norm=self.pc_scale)
        center = (
            feat_dict["center"].reshape(batch_size, self.obs_horizon, 1, 3)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        )
        scale = feat_dict["scale"].reshape(batch_size, self.obs_horizon, 1, 1)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        pc_feat = feat_dict["so3"]  
        obs_cond_vec = pc_feat.reshape(batch_size, -1, 3)


        # proc grasp pose. first split flattened pose to xyz and dir, then normalize xyz. 
        grasp_xyz_raw, grasp_dir1, grasp_dir2 = self.actor._convert_trans_to_vec(grasp_pose)
        grasp_xyz = self.grasp_xyz_normalizer.normalize(grasp_xyz_raw)
        grasp_xyz = (grasp_xyz - center)/scale
        gt_grasp_z = torch.cat([grasp_xyz, grasp_dir1, grasp_dir2], dim=-2)

        # scalar
        left_jpose = self.jpose_normalizer.normalize(left_jpose)
        scalar_left_jpose = self.actor._convert_jpose_to_vec(left_jpose)
        right_jpose = self.jpose_normalizer.normalize(right_jpose)
        scalar_right_jpose = self.actor._convert_jpose_to_vec(right_jpose)
        

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
        
        ## z_t
        left_jpose_noise = torch.randn_like(scalar_left_jpose, device=self.device)
        ## x_t = add_noise(x_0, z_t)
        noisy_left_jpose = self.actor.noise_scheduler.add_noise(
            scalar_left_jpose, left_jpose_noise, timesteps
        )
        right_jpose_noise = torch.randn_like(scalar_right_jpose, device=self.device)
        noisy_right_jpose = self.actor.noise_scheduler.add_noise(
            right_jpose_noise, scalar_right_jpose, timesteps
        )

        vec_grasp_noise = torch.randn_like(gt_grasp_z, device=self.device)  
        noisy_grasp = self.actor.noise_scheduler.add_noise(
            gt_grasp_z, vec_grasp_noise, timesteps
        ) 
        ## /tilde{z}_t = prednet(x_t, Cond, t)
        _, left_scalar_noise_pred = self.actor.left_noise_pred_net_handle(
            noisy_grasp,
            timesteps,
            scalar_sample = noisy_left_jpose, 
            cond=obs_cond_vec,
            scalar_cond=None,
        )

        left_scalar_loss = nn.functional.mse_loss(left_scalar_noise_pred, left_jpose_noise)
        metrics["left_scalar_loss"] = left_scalar_loss


        right_vec_noise_pred, right_scalar_noise_pred = self.actor.right_noise_pred_net_handle(
            noisy_grasp,
            timesteps,
            scalar_sample = noisy_right_jpose, 
            cond=obs_cond_vec,
            scalar_cond=None,
        )
        vec_loss= nn.functional.mse_loss(right_vec_noise_pred, vec_grasp_noise)
        metrics["vec_loss"] = vec_loss
        right_scalar_loss = nn.functional.mse_loss(right_scalar_noise_pred, right_jpose_noise)
        metrics["right_scalar_loss"] = right_scalar_loss

        total_loss = left_scalar_loss + right_scalar_loss + vec_loss
        metrics["total_loss"] = total_loss

        if torch.isnan(total_loss):
            print(f"Loss is nan, please investigate.")
            import pdb

            pdb.set_trace()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.actor.step_ema()

        return metrics
    

    def act(self, obs, history_bid = -1):
        self.train(False)
        assert isinstance(obs["right_pc"][0], np.ndarray)


        batch_size = obs["right_pc"].shape[0]
        # batch_size = 1  # only support batch size 1 for now
        assert history_bid < batch_size # batch to select as denoising history

        xyzs = []

        for batch_idx in range(obs['right_pc'].shape[0]):
            for horizon_id in range(obs['right_pc'].shape[1]):
                xyz = obs['right_pc'][batch_idx][horizon_id]
                if self.shuffle_pc:
                    choice = np.random.choice(
                        xyz.shape[0], self.num_points, replace=True
                    )
                    xyz = xyz[choice, :]
                    xyzs.append(xyz)
                else:
                    # only input certain amount of points
                    step = xyz.shape[0] // self.num_points
                    xyz = xyz[::step, :][: self.num_points]
                    xyzs.append(xyz)
        batch_pc = np.array(xyzs).reshape(obs['right_pc'].shape[0], obs['right_pc'].shape[1], -1, 3)
        torch_obs = dict(
            right_pc=torch.tensor(batch_pc).to(self.device).float(), 
            right_grasp = torch.tensor(obs['right_grasp']).to(self.device).float(),
            left_jpose = torch.tensor(obs['left_jpose']).to(self.device).float(),
            right_jpose = torch.tensor(obs['right_jpose']).to(self.device).float(),
            )
        denoise_history, metrics = self.actor(torch_obs, history_bid=history_bid)


        return denoise_history, metrics
