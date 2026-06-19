import numpy as np
import torch
from torch import nn

from equibot.policies.utils.norm import Normalizer
from equibot.policies.utils.misc import (
    to_torch,
    rotate_observation,
    to_tensor,
    EQUIBOT_PATH,
    ascii_tensor_to_str,
    matched_actions_from_json,
    rotation_surrogate_loss,
    geodestDist,
)
from equibot.policies.utils.diffusion.lr_scheduler import get_scheduler

from equibot.policies.agents.per_skill_policy import EquiSkillPolicy, BiopSkillPolicy



class EquiSkillAgent(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.dataset_type = cfg.data.dataset.dataset_type
        self._init_actor(self.dataset_type)
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
        # self.num_eef = cfg.env.num_eef
        self.num_eef = self.actor.num_eef
        self.dof = cfg.env.dof
        self.num_points = cfg.data.dataset.num_points
        self.obs_mode = cfg.model.obs_mode
        self.ac_mode = cfg.model.ac_mode
        self.obs_horizon = cfg.model.obs_horizon
        self.pred_horizon = cfg.model.pred_horizon
        self.shuffle_pc = cfg.data.dataset.shuffle_pc

        # Metric-aligned x0 auxiliary loss (off by default -> reproduces base eps-MSE).
        # Optimizes the same quantities the eval reports: position L1 (meters) and a
        # smooth geodesic surrogate for rotation. See learn_unimanual_traj.
        self.lambda_pos = cfg.model.get("lambda_pos", 0.0)
        self.lambda_rot = cfg.model.get("lambda_rot", 0.0)
        self.rot_loss_type = cfg.model.get("rot_loss_type", "chordal")
        self.snr_gamma = cfg.model.get("snr_gamma", 5.0)

        self.all_normalizers = None

    def _init_actor(self, dataset_type):
        if 'jpose' in dataset_type:
            self.actor = BiopSkillPolicy(self.cfg, device=self.cfg.device).to(self.cfg.device)
        elif 'traj' in dataset_type:
            self.actor = EquiSkillPolicy(self.cfg, device=self.cfg.device).to(self.cfg.device)
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        self.actor.ema.averaged_model.to(self.cfg.device)

    
    def get_pc_scale(self, pc_data, ac_scale):
        pc = pc_data.reshape(-1, self.num_points, 3)
        centroid = pc.mean(1, keepdim=True)
        centered_pc = pc - centroid
        pc_scale = centered_pc.norm(dim=-1).mean()
        normed_pc_scale = pc_scale / ac_scale
        return normed_pc_scale

    def get_xyz_normalizer(self, xyz_data):
        flattend_xyz = xyz_data.view(-1, 3)
        indices = [[0,1,2]]
        xyz_normalizer = Normalizer(flattend_xyz, symmetric=True, indices=indices)
        return xyz_normalizer

    ## NOTE: only for unimanual
    def _init_multi_normalizers(self, n_data_dict):
        all_normalizers = {}

            ## pc normalizer
        pc_normalizer = self.get_xyz_normalizer(n_data_dict['pc'])
        all_normalizers['pc'] = pc_normalizer
        self.actor.statistics['pc_scale'] = self.get_pc_scale(n_data_dict['pc'], pc_normalizer.stats["max"].max())

        ## skill normalizer
        grasp_normalizer = Normalizer(
            {
                "min": pc_normalizer.stats["min"],
                "max": pc_normalizer.stats["max"],
            }
        )
        all_normalizers['eefpos'] = grasp_normalizer

        gripper_normalizer = Normalizer(n_data_dict['gripper'], symmetric=True, indices=[[0]])
        all_normalizers['gripper'] = gripper_normalizer

        return all_normalizers
    ## max: 0.1598; min: 0
    
    def train(self, training=True):
        self.actor.nets.train(training)



    def learn_unimanual_traj(self,  batch):
        n_data_dict = {}
        # n_data_dict['skill_name_emb'] = batch['skill_name_emb']
        n_data_dict['skill_name'] = batch['skill_name']
        n_data_dict['task_name'] = batch['task_name']
        n_data_dict['eefpos'] = batch['eefpos']
        n_data_dict['gripper'] = batch['gripper']
        # n_data_dict['skill_name'] = ascii_tensor_to_str(batch['skill_name'])

        n_data_dict['pc'] = batch['pc'].repeat(1, self.obs_horizon, 1, 1)
        obs_vec,  center, scale = self.actor.proc_pc(n_data_dict['pc'])

        obs_vec = self.actor.fuse_in_hand_pc(batch, obs_vec, self.actor.fuse_inv_feat)

        eefpos = n_data_dict['eefpos']
        gripper = n_data_dict['gripper']
        # bert_emb = n_data_dict['skill_name_emb']
 
        skill_name_batch, task_name_batch = self.actor.skill_task_ascii_to_str(n_data_dict)

        ## proc grasp
        gt_eefpos_z = self.actor.eef_proc_fn(eefpos, 'eefpos', center, scale)

        ## proc gripper
        gt_gripper_action = self.actor.proc_gripper(gripper, 'gripper')

        batch_size = eefpos.shape[0]
        timesteps = torch.randint(
            0,
            self.actor.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        
        ## z_t
        ## x_t = add_noise(x_0, z_t)

        eefpos_noise = torch.randn_like(gt_eefpos_z, device=self.device)
        noisy_eefpos = self.actor.noise_scheduler.add_noise(gt_eefpos_z, eefpos_noise, timesteps)

        gripper_action_noise = torch.randn_like(gt_gripper_action, device=self.device)
        noisy_gripper_action = self.actor.noise_scheduler.add_noise(gt_gripper_action, gripper_action_noise, timesteps)

        ## /tilde{z}_t = prednet(x_t, Cond, t)
        policy_key = 'unitraj_noise_pred_net'
        
        # skill_scalar_id = self.actor.encode_bert_emb(bert_emb, batch_size)
        task_skill_condition = self.actor.get_all_embs(skill_name_batch, batch_size, task_name_batch)

        # obs_vec = equiv_feat
        # task_skill_condition = torch.cat([inv_feat, task_skill_condition], dim=-1)

        eefpos_noise_pred, gripper_noise_pred = self.actor.nets[policy_key](
            noisy_eefpos,
            timesteps,
            scalar_sample = noisy_gripper_action,
            cond = obs_vec,
            scalar_cond = task_skill_condition,
        )
        
        vec_loss = nn.functional.mse_loss(eefpos_noise_pred, eefpos_noise)
        scalar_loss = nn.functional.mse_loss(gripper_noise_pred, gripper_action_noise)

        aux_metrics = {}
        if self.lambda_pos > 0 or self.lambda_rot > 0:
            vec_loss, aux_metrics = self._add_metric_aligned_loss(
                vec_loss,
                eefpos_noise_pred=eefpos_noise_pred,
                noisy_eefpos=noisy_eefpos,
                timesteps=timesteps,
                center=center,
                scale=scale,
                gt_eefpos=eefpos,
            )

        return vec_loss, scalar_loss, aux_metrics

    def _add_metric_aligned_loss(
        self, vec_loss, *, eefpos_noise_pred, noisy_eefpos, timesteps, center, scale, gt_eefpos
    ):
        """
        Add an x0-space auxiliary loss that optimizes the eval metrics directly:
        position L1 (meters) and a smooth geodesic surrogate for rotation.

        The predicted clean sample is reconstructed from the eps-prediction
        (x0 = (x_t - sqrt(1-ab) * eps) / sqrt(ab)) and pushed through the same
        differentiable recovery as eval, then compared against the ground-truth pose.
        A min-SNR-style weight downweights high-noise timesteps where the single-step
        x0 estimate is ill-conditioned. The base eps-MSE is preserved, which keeps the
        diffusion variable well-determined (Gram-Schmidt is many-to-one).
        """
        alphas_cumprod = self.actor.noise_scheduler.alphas_cumprod.to(self.device)
        ab = alphas_cumprod[timesteps]  # (B,)
        ab = ab.view(-1, *([1] * (noisy_eefpos.dim() - 1)))  # broadcast over (H, D, 3)

        # Reconstruct the predicted clean sample from the eps-prediction.
        x0_pred = (noisy_eefpos - torch.sqrt(1.0 - ab) * eefpos_noise_pred) / torch.sqrt(ab)

        # Recover to world frame exactly as eval does (recovery is differentiable).
        trans_pred, _, _ = self.actor.eef_recover_fn(x0_pred, scale, center, key="eefpos")

        gt_xyz, pred_xyz = gt_eefpos[:, :, :3, 3], trans_pred[:, :, :3, 3]
        gt_R, pred_R = gt_eefpos[:, :, :3, :3], trans_pred[:, :, :3, :3]

        # Per-sample losses (B,), then min-SNR weighting (snr = ab / (1 - ab)).
        pos_per_sample = (pred_xyz - gt_xyz).abs().flatten(1).mean(1)
        rot_per_sample = rotation_surrogate_loss(pred_R, gt_R, self.rot_loss_type).flatten(1).mean(1)

        ab_flat = ab.flatten()
        snr = ab_flat / (1.0 - ab_flat).clamp(min=1e-8)
        snr_weight = torch.clamp(snr / self.snr_gamma, max=1.0)  # (B,) in (0, 1]

        pos_loss = (snr_weight * pos_per_sample).mean()
        rot_loss = (snr_weight * rot_per_sample).mean()
        vec_loss = vec_loss + self.lambda_pos * pos_loss + self.lambda_rot * rot_loss

        # Monitoring only -- keys must NOT end in '_loss' or they get summed into total_loss.
        with torch.no_grad():
            rot_deg = geodestDist(gt_R, pred_R).mean() * 180.0 / torch.pi
        aux_metrics = {
            "aux_pos_l1_m": pos_per_sample.mean().detach(),
            "aux_rot_surrogate": rot_per_sample.mean().detach(),
            "aux_rot_diff_deg": rot_deg.detach(),
        }
        return vec_loss, aux_metrics


    def learn_bimanual_jpose(self,  batch):
        n_data_dict = {}
        n_data_dict['skill_name'] = batch['skill_name']
        n_data_dict['task_name'] = batch['task_name']
        n_data_dict['jpose'] = batch['jpose']

        jpose_key = 'jpose'
        scalar_dual_jpose_raw = n_data_dict[jpose_key]
        batch_size = scalar_dual_jpose_raw.shape[0]
        scalar_dual_jpose_raw = scalar_dual_jpose_raw.reshape(batch_size, -1, self.dof)
        scalar_dual_jpose = self.actor.proc_jpose(scalar_dual_jpose_raw,  jpose_key).squeeze(1)
        timesteps = torch.randint(
            0,
            self.actor.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        jpose_noise = torch.randn_like(scalar_dual_jpose, device=self.device)
        noisy_jpose = self.actor.noise_scheduler.add_noise(
            scalar_dual_jpose, jpose_noise, timesteps
        )

        scalar_noise_pred = self.actor.nets['jpose_noise_pred_net'](noisy_jpose, timesteps)
        scalar_loss = nn.functional.mse_loss(scalar_noise_pred, jpose_noise)
        return scalar_loss


    def update(self, batch):
        self.train()

        ###### Load data, preprocessing using mask ######
        batch = to_torch(batch, self.device)

        ## assume the data is already normalized
        if self.all_normalizers is None and self.cfg.data.dataset.normalization_method == "batch":
            self.all_normalizers = self._init_multi_normalizers(n_data_dict)
            self.actor.all_normalizers = self.all_normalizers


    ######## train the pred net ########
        metrics = {}
        if 'jpose' in self.dataset_type:
            scalar_loss = self.learn_bimanual_jpose(batch)
        elif 'traj' in self.dataset_type:
            vec_loss, scalar_loss, traj_aux = self.learn_unimanual_traj(batch)
            metrics['vec_loss'] = vec_loss
            metrics.update(traj_aux)
        else:
            raise ValueError(f"Invalid dataset type: {self.dataset_type}")
        metrics['scalar_loss'] = scalar_loss

        total_loss = 0
        for metric_key in metrics:
            if metric_key.endswith('_loss'):
                total_loss += metrics[metric_key]

        metrics["log_loss"] = np.log(total_loss.detach().cpu().numpy())

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

    def set_normalizer_and_statistics(self, dataset):
        self.actor.statistics = dataset.statistics
        self.actor.skill_names = dataset.skill_names
        self.actor.task_names = dataset.task_names

        if self.cfg.data.dataset.normalization_method == "all":
            self.set_normalizer(dataset.normalizer.state_dict())

    def set_normalizer(self, normalizer_state_dict):
        self.actor.normalizer.load_state_dict(normalizer_state_dict)


    def fix_checkpoint_keys(self, state_dict):
        fixed_state_dict = dict()
        for k, v in state_dict.items():
            if "encoder.encoder" in k:
                fixed_k = k.replace("encoder.encoder", "encoder")
            else:
                fixed_k = k
            if "handle" in k:
                continue
            fixed_state_dict[fixed_k] = v
        return fixed_state_dict
    
    def save_snapshot(self, save_path):
        state_dict = dict(
            cfg = self.cfg,
            actor=self.actor.state_dict(),
            ema_model=self.actor.ema.averaged_model.state_dict(),
        )
        # state_dict['skill_name_to_emb_tensor'] = self.actor.skill_name_to_emb_tensor
        
        state_dict["statistics"] = self.actor.statistics

        if self.cfg.data.dataset.normalization_method == "all":
            state_dict["normalizer"] = self.actor.normalizer.state_dict()
            # state_dict["statistics"] = self.actor.statistics
        else:
            state_dict["eefpos_normalizer"] = self.all_normalizers["eefpos"].state_dict()
            state_dict["gripper_normalizer"] = self.all_normalizers["gripper"].state_dict()
            # state_dict["pc_scale"] = self.actor.statistics["pc_scale"]
            state_dict["pc_normalizer"] = self.all_normalizers["pc"].state_dict()

        torch.save(state_dict, save_path)

    def load_state_dict_to_actor(self, state_dict):
        self.actor.cfg = state_dict["cfg"]
        self.cfg = state_dict["cfg"]
    
        self.actor.load_state_dict(self.fix_checkpoint_keys(state_dict["actor"]))
        self.actor._init_torch_compile()

        self.actor.ema.averaged_model.load_state_dict(
            self.fix_checkpoint_keys(state_dict["ema_model"])
        )

        self.actor.statistics = state_dict["statistics"]
        self.actor.skill_names = list(state_dict["statistics"]["skill_embs_all_tasks"].keys())


        if self.cfg.data.dataset.normalization_method == "all":
            self.set_normalizer(state_dict["normalizer"])
        else:

            self.all_normalizers = {}
            self.all_normalizers["eefpos"] =Normalizer(state_dict["eefpos_normalizer"])
            self.all_normalizers["gripper"] = Normalizer(state_dict["gripper_normalizer"])
            self.all_normalizers["pc"] = Normalizer(state_dict["pc_normalizer"])
            self.actor.all_normalizers = self.all_normalizers




    def load_snapshot(self, load_path):
        import os
        load_path_full = os.path.join(EQUIBOT_PATH, load_path)
        state_dict = torch.load(load_path_full)
        self.load_state_dict_to_actor(state_dict)

            
    ## call this function during evaluation (only during training)
    def eval_with_rotation(self, obs, skill_id = -1):
        self.train(False)
        random_yaw = np.random.uniform(-np.pi, np.pi)
        np_obs = rotate_observation(obs, random_yaw)
        cpu_obs = to_tensor(np_obs)
        gpu_obs = to_torch(cpu_obs, self.device)

        # gpu_obs = obs
        with torch.no_grad():
            action_dict, eval_metrics, denoise_history = self.actor(gpu_obs, skill_id=skill_id)

        return denoise_history, eval_metrics


    
    def get_skillwise_sgs_from_statistics(self, skill_name):
        action_sgs_json = self.actor.statistics['matched_action_sgs'][skill_name]
        import json
        action_sgs = json.loads(action_sgs_json)
        skill_info_nx = matched_actions_from_json(action_sgs)
        return skill_info_nx
