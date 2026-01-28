"""
SDP Agent for training and inference with Spherical Diffusion Policy.

Wraps SDPPolicy with training loop, normalization, and checkpoint management.
Based on EquiSkillAgent structure for compatibility with existing training scripts.
"""

import numpy as np
import torch
from torch import nn

from equibot.policies.utils.norm import Normalizer
from equibot.policies.utils.misc import to_torch, rotate_observation, to_tensor, EQUIBOT_PATH
from equibot.policies.utils.diffusion.lr_scheduler import get_scheduler
from equibot.policies.agents.sdp_policy import SDPPolicy


class SDPAgent:
    """
    Agent wrapper for SDP policy with training/inference logic.
    
    Handles:
    - Training loop with diffusion loss
    - Normalization management
    - Checkpoint save/load
    - EMA updates
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        
        # Initialize SDP policy
        self.actor = SDPPolicy(cfg, device=cfg.device).to(cfg.device)
        self.actor.ema.averaged_model.to(cfg.device)
        
        # Training setup
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
        
        # Config parameters
        # self.num_eef = self.actor.num_eef
        # self.dof = cfg.env.dof
        self.num_points = cfg.data.dataset.num_points
        self.obs_mode = cfg.model.obs_mode
        self.ac_mode = cfg.model.ac_mode
        self.obs_horizon = cfg.model.obs_horizon
        self.pred_horizon = cfg.model.pred_horizon
        self.shuffle_pc = cfg.data.dataset.shuffle_pc
        
        self.all_normalizers = None
    
    def train(self, training=True):
        """Set training mode."""
        self.actor.nets.train(training)
    
    def get_pc_scale(self, pc_data, ac_scale):
        """Compute point cloud scale for normalization."""
        pc = pc_data.reshape(-1, self.num_points, 3)
        centroid = pc.mean(1, keepdim=True)
        centered_pc = pc - centroid
        pc_scale = centered_pc.norm(dim=-1).mean()
        normed_pc_scale = pc_scale / ac_scale
        return normed_pc_scale
    
    def get_xyz_normalizer(self, xyz_data):
        """Create XYZ normalizer."""
        flattend_xyz = xyz_data.view(-1, 3)
        indices = [[0, 1, 2]]
        xyz_normalizer = Normalizer(flattend_xyz, symmetric=True, indices=indices)
        return xyz_normalizer
    
    def _init_multi_normalizers(self, n_data_dict):
        """Initialize normalizers for different data modalities."""
        all_normalizers = {}
        
        # PC normalizer
        pc_normalizer = self.get_xyz_normalizer(n_data_dict['pc'])
        all_normalizers['pc'] = pc_normalizer
        self.actor.statistics['pc_scale'] = self.get_pc_scale(
            n_data_dict['pc'], pc_normalizer.stats["max"].max()
        )
        
        # EEF normalizer (use PC normalizer stats)
        grasp_normalizer = Normalizer({
            "min": pc_normalizer.stats["min"],
            "max": pc_normalizer.stats["max"],
        })
        all_normalizers['eefpos'] = grasp_normalizer
        
        # Gripper normalizer
        gripper_normalizer = Normalizer(n_data_dict['gripper'], symmetric=True, indices=[[0]])
        all_normalizers['gripper'] = gripper_normalizer
        
        return all_normalizers
    
    def learn_unimanual_traj(self, batch):
        """
        Learn trajectory prediction using diffusion loss.
        
        Following per_skill_agent.learn_unimanual_traj pattern.
        
        Training objective (from paper):
        L = ||epsilon_theta(S_t, A_t^0 + epsilon, k) - epsilon||^2
        
        IrrepConditionalUnet1D expects sample: (B, T, 10) = 9D pos/rot + 1D gripper
        """
        n_data_dict = {}
        n_data_dict['skill_name'] = batch['skill_name']
        n_data_dict['task_name'] = batch['task_name']
        n_data_dict['eefpos'] = batch['eefpos']
        n_data_dict['gripper'] = batch['gripper']
        

        # Get text conditioning (following per_skill pattern)
        skill_name_batch, task_name_batch = self.actor.skill_task_ascii_to_str(n_data_dict)
        task_skill_condition = self.actor.get_all_embs(skill_name_batch, len(skill_name_batch), task_name_batch)
        

        # Process point cloud through SDPEncoder (following per_skill pattern)
        n_data_dict['pc'] = batch['pc'].repeat(1, self.obs_horizon, 1, 1)
        global_cond, center, scale = self.actor.proc_pc(n_data_dict['pc'], language_emb=task_skill_condition)
        
        eefpos = n_data_dict['eefpos']
        gripper = n_data_dict['gripper']
        
        
        # Process ground truth actions (following per_skill pattern)
        gt_eefpos_z = self.actor.eef_proc_fn(eefpos, 'eefpos', center, scale)
        gt_gripper_action = self.actor.proc_gripper(gripper, 'gripper')
        
        # Combine into sample format for IrrepConditionalUnet1D: (B, T, 10)
        gt_sample = self.actor.combine_eef_gripper(gt_eefpos_z, gt_gripper_action)
        
        batch_size = eefpos.shape[0]
        
        # Sample random timesteps (following per_skill pattern)
        timesteps = torch.randint(
            0,
            self.actor.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        
        # Add noise to ground truth sample
        noise = torch.randn_like(gt_sample, device=self.device)
        noisy_sample = self.actor.noise_scheduler.add_noise(gt_sample, noise, timesteps)
        
        # Predict noise using SDTU
        policy_key = 'unitraj_noise_pred_net'
        noise_pred = self.actor.nets[policy_key](
            sample=noisy_sample,
            timestep=timesteps,
            global_cond=global_cond,
        )
        
        # Compute losses (separate vec and scalar for logging, following per_skill pattern)
        # First 9 dims are eef (vectors), last 1 dim is gripper (scalar)
        vec_loss = nn.functional.mse_loss(noise_pred[..., :9], noise[..., :9])
        scalar_loss = nn.functional.mse_loss(noise_pred[..., 9:], noise[..., 9:])
        
        return vec_loss, scalar_loss
    
    def update(self, batch):
        """
        Update model with one training step.
        
        Following per_skill_agent.update pattern.
        
        Returns:
            dict: Training metrics
        """
        self.train()
        
        # Move batch to device
        batch = to_torch(batch, self.device)
        
        # Initialize normalizers on first batch if needed (following per_skill pattern)
        if self.all_normalizers is None and self.cfg.data.dataset.normalization_method == "batch":
            n_data_dict = {
                'pc': batch['pc'],
                'gripper': batch['gripper'],
            }
            self.all_normalizers = self._init_multi_normalizers(n_data_dict)
            self.actor.all_normalizers = self.all_normalizers
        
        # Train trajectory prediction (following per_skill pattern)
        metrics = {}
        vec_loss, scalar_loss = self.learn_unimanual_traj(batch)
        metrics['vec_loss'] = vec_loss
        metrics['scalar_loss'] = scalar_loss
        
        # Compute total loss (following per_skill pattern)
        total_loss = 0
        for metric_key in metrics:
            if metric_key.endswith('_loss'):
                total_loss += metrics[metric_key]
        
        metrics["log_loss"] = np.log(total_loss.detach().cpu().numpy())
        
        if torch.isnan(total_loss):
            print("Loss is nan, please investigate.")
            import pdb
            pdb.set_trace()
        
        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        
        # EMA update
        self.actor.step_ema()
        
        return metrics
    
    def set_normalizer_and_statistics(self, dataset):
        """Set normalizer and statistics from dataset."""
        self.actor.statistics = dataset.statistics
        self.actor.skill_names = dataset.skill_names
        self.actor.task_names = dataset.task_names
        
        if self.cfg.data.dataset.normalization_method == "all":
            self.set_normalizer(dataset.normalizer.state_dict())
    
    def set_normalizer(self, normalizer_state_dict):
        """Load normalizer state dict."""
        self.actor.normalizer.load_state_dict(normalizer_state_dict)
    
    def fix_checkpoint_keys(self, state_dict):
        """Fix checkpoint keys for compatibility."""
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
        """Save model checkpoint."""
        state_dict = dict(
            cfg=self.cfg,
            actor=self.actor.state_dict(),
            ema_model=self.actor.ema.averaged_model.state_dict(),
        )
        state_dict["statistics"] = self.actor.statistics
        
        if self.cfg.data.dataset.normalization_method == "all":
            state_dict["normalizer"] = self.actor.normalizer.state_dict()
        else:
            if self.all_normalizers is not None:
                state_dict["eefpos_normalizer"] = self.all_normalizers["eefpos"].state_dict()
                state_dict["gripper_normalizer"] = self.all_normalizers["gripper"].state_dict()
                state_dict["pc_normalizer"] = self.all_normalizers["pc"].state_dict()
        
        torch.save(state_dict, save_path)
    
    def load_state_dict_to_actor(self, state_dict):
        """Load state dict to actor (following per_skill pattern)."""
        self.actor.cfg = state_dict["cfg"]
        self.cfg = state_dict["cfg"]
        
        self.actor.load_state_dict(self.fix_checkpoint_keys(state_dict["actor"]))
        self.actor._init_torch_compile()
        
        self.actor.ema.averaged_model.load_state_dict(
            self.fix_checkpoint_keys(state_dict["ema_model"])
        )
        
        self.actor.statistics = state_dict["statistics"]
        self.actor.skill_names = list(state_dict["statistics"]["skill_embs_all_tasks"].keys())
        self.actor.task_names = list(state_dict["statistics"]["task_emb_dict"].keys())
        
        if self.cfg.data.dataset.normalization_method == "all":
            self.set_normalizer(state_dict["normalizer"])
        else:
            if "eefpos_normalizer" in state_dict:
                self.all_normalizers = {}
                self.all_normalizers["eefpos"] = Normalizer(state_dict["eefpos_normalizer"])
                self.all_normalizers["gripper"] = Normalizer(state_dict["gripper_normalizer"])
                self.all_normalizers["pc"] = Normalizer(state_dict["pc_normalizer"])
                self.actor.all_normalizers = self.all_normalizers
    
    def load_snapshot(self, load_path):
        """Load model from checkpoint."""
        import os
        load_path_full = os.path.join(EQUIBOT_PATH, load_path)
        state_dict = torch.load(load_path_full)
        self.load_state_dict_to_actor(state_dict)
    
    def eval_with_rotation(self, obs, skill_id=-1):
        """Evaluate with random rotation augmentation."""
        self.train(False)
        random_yaw = np.random.uniform(-np.pi, np.pi)
        np_obs = rotate_observation(obs, random_yaw)
        cpu_obs = to_tensor(np_obs)
        gpu_obs = to_torch(cpu_obs, self.device)
        
        with torch.no_grad():
            action_dict, eval_metrics, denoise_history = self.actor(gpu_obs, skill_id=skill_id)
        
        return denoise_history, eval_metrics

