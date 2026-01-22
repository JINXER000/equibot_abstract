"""
Spherical Diffusion Policy (SDP) for object-centric trajectory prediction.

Reference: paper_ref/sdp/example_paper.tex

Architecture:
1. EquiformerV2 Encoder - Point cloud to spherical Fourier features (SO(3) equivariant)
2. Language Encoder - Text conditioning for skill/task
3. SDTU (Spherical Denoising Temporal U-net) - Noise estimation with SFiLM conditioning

Key equations from paper:
- Mixing channel temporal convolution (Eq. 4):
  h_{l,m,t}^{o} = sum_j sum_{i in in} h_{l,m,j}^{i} w_{l,j-t}^{i,o}
  
- Spherical FiLM conditioning (Eq. 6):
  SFiLM(h_l | gamma_l, beta_l) = gamma_l^T h_l (h_l / ||h_l||) + beta_l

- Action representation: rho_ee = rho_1^4 + rho_0
  (4 type-1 vectors for pos + 3 rot columns, 1 scalar for gripper)

IrrepConditionalUnet1D interface:
- Input sample: (B, T, 10) - 9D for pos/rot (3 vectors x 3) + 1D for gripper
- Output: (B, T, 10) - same format (noise prediction)
- global_cond: (B, C * irrep_dim) - flattened spherical conditioning
"""

import copy
import hydra
import torch
from torch import nn
import numpy as np
import einops

from equibot.policies.vision.sdp_encoder import SDPEncoder
from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.sdp_diffusion.irreps_conditional_unet1d import IrrepConditionalUnet1D
from equibot.policies.utils.normalizer import LinearNormalizer

from equibot.policies.utils.lan_utils import MLPEncoder

from equibot.policies.utils.misc import (
    to_torch, convert_trans_to_vec, convert_vec_to_trans,
    geodestDist, convert_trans_to_4pts, convert_4pts_to_trans,
    matrix_to_rotation_6d, render_trajectory, ascii_tensor_batch_to_str
)


class SDPPolicy(nn.Module):
    """
    Spherical Diffusion Policy for object-centric trajectory prediction.
    
    Inputs (from per_skill_dataset):
        - pc: Point cloud [B, 1, N, 3] (first-sight observation)
        - skill_name: Text embedding for skill conditioning
        - task_name: Text embedding for task conditioning
    
    Outputs:
        - eefpos: SE(3) trajectory [B, T, 4, 4]
        - gripper: Gripper actions [B, T, 1]
        
    Key differences from EquiBot (Vector Neurons):
        - Uses spherical harmonics (degree L) vs vector neurons (degree 1)
        - Higher expressiveness: (L+1)^2 coefficients vs 3D vectors
        - SFiLM conditioning vs VecLNA conditioning
    """
    
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        
        self.cfg = cfg
        self.device = device
        self.use_torch_compile = cfg.model.use_torch_compile
        
        # Horizons (following reference: horizon, n_action_steps, n_obs_steps)
        self.pred_horizon = cfg.model.pred_horizon
        self.obs_horizon = cfg.model.obs_horizon
        self.action_horizon = cfg.model.ac_horizon
        self.horizon = self.pred_horizon  # Alias for reference compatibility
        self.n_action_steps = self.action_horizon
        self.n_obs_steps = self.obs_horizon
        
        # Diffusion parameters
        if hasattr(cfg.model, "num_diffusion_iters"):
            self.num_diffusion_iters = cfg.model.num_diffusion_iters
        else:
            self.num_diffusion_iters = cfg.model.noise_scheduler.num_train_timesteps
        
        # Environment parameters
        self.dof = cfg.env.dof
        self.num_eef = cfg.env.num_eef
        
        # Policy parameters (following reference step.py)
        self.canonicalize = cfg.model.get('canonicalize', True)
        self.use_pc_color = cfg.data.dataset.get('use_pc_color', False)
        self.obs_as_global_cond = cfg.model.get('obs_as_global_cond', True)
        self.condition_type = cfg.model.get('condition_type', 'film')
        self.denoise_nn = cfg.model.get('denoise_nn', 'irrep')
        self.norm = cfg.model.get('norm', False)
        self.rot_aug = cfg.model.get('rot_aug', None)
        self.rad_aug = cfg.model.get('rad_aug', 0)
        self.pcd_noise = cfg.model.get('pcd_noise', 0)
        self.n_groups = cfg.model.get('n_groups', 8)
        
        # Spherical harmonic parameters
        encoder_cfg = cfg.model.encoder
        self.lmax = encoder_cfg.lmax
        self.mmax = encoder_cfg.get('mmax', encoder_cfg.lmax)
        self.irrep_dim = (self.lmax + 1) ** 2
        
        # Normalizers
        self.normalizer = LinearNormalizer()
        self.statistics = {}
        
        # EEF representation - calculate action_dim following reference pattern
        # For 3vec: 3 vectors (3D each) + 1 gripper = 10D total
        # For 4pts: 4 keypoints (3D each) + 1 gripper = 13D total
        self.eef_representation = cfg.data.dataset.eef_representation
        if self.eef_representation == "3vec":
            self.eef_proc_fn = self.proc_eef_3vec
            self.eef_recover_fn = self.recover_eef_3vec
            self.eef_dims = 3  # 3 vectors (pos + 2 rotation vectors)
            self.action_dim = 10  # 3*3 + 1 = 9D vectors + 1D gripper
        elif self.eef_representation == "4pts":
            self.eef_proc_fn = self.proc_eef_4pts
            self.eef_recover_fn = self.recover_eef_4pts
            self.original_gripper_pcd = np.array(cfg.data.dataset.original_gripper_pcd)
            self.eef_dims = 4  # 4 keypoints
            self.action_dim = 13  # 4*3 + 1 = 12D keypoints + 1D gripper
        else:
            raise ValueError(f"Unsupported eef_representation: {self.eef_representation}")
        
        # Build networks
        net_dict = {}
        
        # 1. SDPEncoder (EquiformerV2) for point cloud - following reference initialization
        encoder_output_dim = encoder_cfg.get('encoder_output_dim', encoder_cfg.c_dim)
        use_color = cfg.model.get('use_pc_color', False)  
        
        # Language encoder config (needed for encoder language fusion)
        language_encoder_cfg = cfg.model.language_encoder_cfg
        language_embed_dim = language_encoder_cfg.hidden_size * 2  # skill + task embeddings
        
        # Language fusion dimension: use hidden_size from language encoder (can be overridden in encoder_cfg)
        language_fusion_dim = encoder_cfg.get('language_fusion_dim', language_encoder_cfg.hidden_size)
        
        net_dict['obj_encoder'] = SDPEncoder(
            c_dim=encoder_output_dim,
            use_color=use_color,
            language_embed_dim=language_embed_dim,  # Input: skill + task embeddings (hidden_size * 2)
            language_fusion_dim=language_fusion_dim,  # Output: fused dimension (defaults to hidden_size)
            lmax=self.lmax,
            mmax=self.mmax,
            max_neighbors=tuple(encoder_cfg.get('max_neighbors', (16, 16, 16, 16))),
            max_radius=tuple(encoder_cfg.get('max_radius', (0.05, 0.2, 0.8, 3))),
            pool_ratio=tuple(encoder_cfg.get('pool_ratio', (0.25, 0.25, 0.25))),
            sphere_channels=tuple(encoder_cfg.get('sphere_channels', (32, 64, 128))),
            attn_hidden_channels=tuple(encoder_cfg.get('attn_hidden_channels', (32, 64, 128, 256))),
            attn_alpha_channels=tuple(encoder_cfg.get('attn_alpha_channels', (8, 16, 32, 64))),
            attn_value_channels=tuple(encoder_cfg.get('attn_value_channels', (4, 8, 16, 32))),
            ffn_hidden_channels=tuple(encoder_cfg.get('ffn_hidden_channels', (32, 64, 128, 256))),
            edge_channels=tuple(encoder_cfg.get('edge_channels', (16, 32, 64, 128))),
            num_distance_basis=tuple(encoder_cfg.get('num_distance_basis', (64, 64, 64, 64))),
            num_heads=encoder_cfg.get('num_heads', 4),
            pcd_noise=self.pcd_noise,
            norm=self.norm,
            deterministic=encoder_cfg.get('deterministic', False),
            grid_resolution=encoder_cfg.get('v_grid_resolution', encoder_cfg.get('grid_resolution', 12)),
            pool_method=encoder_cfg.get('pool_method', 'fpsknn'),
            alpha_drop=encoder_cfg.get('alpha_drop', 0.1),
            drop_path_rate=encoder_cfg.get('drop_path_rate', 0.0),
            proj_drop=encoder_cfg.get('proj_drop', 0.1),
        )
        
        self.encoder_out_dim = encoder_output_dim
        # Get obs_feature_dim from encoder output_shape (following reference)
        # For SDPEncoder with language fusion, output_dim() returns (c_dim + lang_dim) * irrep_dim
        obs_feature_dim = net_dict['obj_encoder'].output_shape()
        self.obs_feature_dim = obs_feature_dim
        
        # Calculate input_dim and global_cond_dim following reference pattern
        input_dim = self.action_dim + obs_feature_dim
        global_cond_dim = None
        if self.obs_as_global_cond:
            input_dim = self.action_dim
            # Language embeddings are now fused into encoder output, so no need to add separately
            global_cond_dim = obs_feature_dim * self.obs_horizon
        
        # 2. Language encoder for text conditioning (used for encoder fusion)
        language_encoder_cfg = cfg.model.language_encoder_cfg
        output_size = language_encoder_cfg.hidden_size
        self.language_encoder = self._setup_language_encoder(output_size=output_size, **language_encoder_cfg)
        net_dict['language_encoder'] = self.language_encoder
        
        # 3. SDTU (Spherical Denoising Temporal U-net) - following reference
        diffusion_cfg = cfg.model.get('diffusion', {})
        d_grid_resolution = encoder_cfg.get('d_grid_resolution', encoder_cfg.get('grid_resolution', 14))
        
        if self.denoise_nn == 'irrep':
            # Following reference: input_dim // 3 because IrrepConditionalUnet1D expects type-1 channels
            # For 3vec: 10 // 3 = 3 (3 vectors), for 4pts: 13 // 3 = 4 (4 keypoints as vectors)
            model = IrrepConditionalUnet1D(
                input_dim=input_dim // 3,
                max_lmax=self.lmax,
                norm=self.norm,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_cfg.get('diffusion_step_embed_dim', 128),
                down_dims=diffusion_cfg.get('down_dims', [200, 400, 800]),
                kernel_size=diffusion_cfg.get('kernel_size', 5),
                n_groups=self.n_groups,
                FiLM_type=diffusion_cfg.get('FiLM_type', 'SFiLM'),
                grid_resolution=d_grid_resolution
            )
        else:
            raise NotImplementedError(f"denoise_nn={self.denoise_nn} not implemented (only 'irrep' supported)")
        
        net_dict['unitraj_noise_pred_net'] = model
        
        self.nets = nn.ModuleDict(net_dict)
        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)
        
        self._init_torch_compile()
        
        # Noise scheduler
        self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler)
        
        # Print parameter counts (following reference)
        diffusion_params = sum(p.numel() for p in self.nets['unitraj_noise_pred_net'].parameters())
        vision_params = sum(p.numel() for p in self.nets['obj_encoder'].parameters())
        print(f"Diffusion params: {diffusion_params:.2e}")
        print(f"Vision params: {vision_params:.2e}")
        print(f"[SDPPolicy] use_pc_color: {self.use_pc_color}")
        print(f"[SDPPolicy] canonicalize: {self.canonicalize}")
        print(f"[SDPPolicy] obs_as_global_cond: {self.obs_as_global_cond}")
    
    def _init_torch_compile(self):
        if self.use_torch_compile:
            self.nets = nn.ModuleDict({
                key: torch.compile(net) for key, net in self.nets.items()
            })
    
    def _setup_language_encoder(self, network_name, **language_encoder_kwargs):
        return eval(network_name)(**language_encoder_kwargs)
    
    def step_ema(self):
        self.ema.step(self.nets)
    
    # ==================== Normalization ====================
    def normalize_from_key(self, key, data):
        if self.cfg.data.dataset.normalization_method == "batch":
            return self.all_normalizers[key].normalize(data)
        else:
            return self.normalizer[key].normalize(data)
    
    def unnormalize_from_key(self, key, data):
        if self.cfg.data.dataset.normalization_method == "batch":
            return self.all_normalizers[key].unnormalize(data)
        else:
            return self.normalizer[key].unnormalize(data)
    
    # ==================== Text Encoding ====================
    def get_encoding_from_name_batch(self, name_batch, batch_size, mapping_dict):
        if batch_size == 1:
            emb_tensor = mapping_dict[name_batch]
            if isinstance(emb_tensor, np.ndarray):
                emb_tensor = torch.tensor(emb_tensor).to(self.device)
            emb_batch = self.encode_bert_emb(emb_tensor, batch_size)
        else:
            emb_tensor_batch = []
            for skill_name in name_batch:
                emb_tensor = mapping_dict[skill_name]
                if isinstance(emb_tensor, np.ndarray):
                    emb_tensor = torch.tensor(emb_tensor).to(self.device)
                emb_tensor_batch.append(emb_tensor)
            emb_tensor_batch = torch.stack(emb_tensor_batch, dim=0)
            emb_batch = self.encode_bert_emb(emb_tensor_batch, batch_size)
        return emb_batch
    
    def encode_bert_emb(self, bert_emb, batch_size):
        skill_emb = self.nets['language_encoder'](bert_emb)
        skill_emb_batch = skill_emb.reshape(batch_size, -1)
        return skill_emb_batch
    
    def get_all_embs(self, skill_name_batch, batch_size, task_name_batch=None):
        skill_emb_batch = self.get_encoding_from_name_batch(
            skill_name_batch, batch_size, self.statistics['skill_embs_all_tasks']
        )
        if task_name_batch is not None:
            task_emb_batch = self.get_encoding_from_name_batch(
                task_name_batch, batch_size, self.statistics['task_emb_dict']
            )
            skill_emb_batch = torch.cat([skill_emb_batch, task_emb_batch], dim=-1)
        return skill_emb_batch
    
    # ==================== Point Cloud Processing ====================
    def proc_pc(self, pc, language_emb=None, ema_nets=None):
        """
        Process point cloud through SDPEncoder with optional language fusion.
        
        Following reference step.py pattern:
        - Strips color if use_pc_color=False
        - Normalizes point cloud
        - Passes through encoder to get spherical features (with language fused)
        - Returns obs_vec (conditioning), center, and scale for canonicalization
        
        Args:
            pc: (B, obs_horizon, N, D) point cloud (D=3 or 6)
            language_emb: Optional (B, language_embed_dim) language embeddings to fuse
            ema_nets: optional EMA networks for inference
            
        Returns:
            obs_vec: (B, obs_horizon * (c_dim + lang_dim) * irrep_dim) spherical conditioning features
            center: (B, pred_horizon, 1, 3) canonicalization center
            scale: (B, pred_horizon, 1, 1) canonicalization scale
        """
        # Strip color if not using it (following reference step.py line 258-259)
        if not self.use_pc_color:
            pc = pc[..., :3]
        
        pc_key = 'pc'
        pc = self.normalize_from_key(pc_key, pc)
        batch_size = pc.shape[0]
        
        encoder_key = 'obj_encoder'
        if ema_nets is None:
            encoder_handle = self.nets[encoder_key]
        else:
            encoder_handle = ema_nets[encoder_key]
        
        pc_scale = self.statistics['pc_scale']
        # Pass language embeddings to encoder for fusion
        feat_dict = encoder_handle(pc, target_norm=pc_scale, language_emb=language_emb)
        
        # Extract outputs
        # s2_feat: [B, obs_horizon, (c_dim + lang_dim) * irrep_dim] (with language fused)
        s2_feat = feat_dict['s2_feat']
        
        # Reshape for conditioning following reference pattern
        if self.obs_as_global_cond:
            if "cross_attention" in self.condition_type:
                # Treat as sequence: (B, obs_horizon, (c_dim + lang_dim) * irrep_dim)
                obs_vec = s2_feat
            else:
                # Flatten: (B, obs_horizon * (c_dim + lang_dim) * irrep_dim)
                obs_vec = s2_feat.reshape(batch_size, -1)
        else:
            # Not used as global cond, return as-is
            obs_vec = s2_feat.reshape(batch_size, -1)
        
        # Get center and scale for action canonicalization (same as per_skill)
        center = feat_dict['center'][:, [-1]].repeat(1, self.pred_horizon, 1, 1)  # [B, pred_horizon, 1, 3]
        scale = feat_dict['scale'][:, [-1]].repeat(1, self.pred_horizon, 1, 1)    # [B, pred_horizon, 1, 1]
        
        return obs_vec, center, scale
    
    # ==================== EEF Processing ====================
    def proc_eef_3vec(self, eef_pose, key, center, scale):
        eef_xyz_raw, eef_dir1, eef_dir2 = convert_trans_to_vec(eef_pose)
        eef_xyz = self.normalize_from_key(key, eef_xyz_raw)
        eef_xyz = (eef_xyz - center) / scale
        gt_eef_z = torch.cat([eef_xyz, eef_dir1, eef_dir2], dim=-2)
        return gt_eef_z
    
    def proc_eef_4pts(self, eef_pose, key, center, scale):
        eef_4pts_raw = convert_trans_to_4pts(eef_pose, self.original_gripper_pcd)
        eef_4pts = self.normalize_from_key(key, eef_4pts_raw)
        eef_4pts = (eef_4pts - center) / scale
        return eef_4pts
    
    def recover_eef_3vec(self, eefpos_batch, scale, center, key):
        scale = torch.mean(scale, dim=2, keepdim=True)
        center = torch.mean(center, dim=2, keepdim=True)
        batch_size = eefpos_batch.shape[0]
        
        eefpos_xyz = eefpos_batch[:, :, 0, :].reshape(batch_size, -1, 1, 3)
        eefpos_xyz = eefpos_xyz * scale + center
        unnormed_eefpos_xyz = self.unnormalize_from_key(key, eefpos_xyz)
        
        rot6d_batch = eefpos_batch[:, :, 1:, :].reshape(batch_size, -1, 1, 6)
        trans_batch = convert_vec_to_trans(rot6d_batch, unnormed_eefpos_xyz)
        
        return trans_batch, unnormed_eefpos_xyz, rot6d_batch
    
    def recover_eef_4pts(self, eefpt_batch, scale, center, key):
        scale = torch.mean(scale, dim=2, keepdim=True)
        center = torch.mean(center, dim=2, keepdim=True)
        
        eefpt_batch_uncano = eefpt_batch * scale + center
        unnormed_eefpt_batch = self.unnormalize_from_key(key, eefpt_batch_uncano)
        
        trans_batch = convert_4pts_to_trans(unnormed_eefpt_batch, self.original_gripper_pcd)
        
        unnormed_eefpos_xyz = trans_batch[:, :, :3, 3:].transpose(-2, -1)
        eef_rot = trans_batch[:, :, :3, :3].reshape(-1, 3, 3)
        rot6d_batch = matrix_to_rotation_6d(eef_rot)
        
        return trans_batch, unnormed_eefpos_xyz, rot6d_batch
    
    def proc_gripper(self, raw_gripper, key):
        gripper_action = self.normalize_from_key(key, raw_gripper)
        gripper_action = gripper_action.reshape(gripper_action.shape[0], -1, 1)
        return gripper_action
    
    def recover_gripper(self, gripper_batch, key):
        unnormed_gripper = (
            self.unnormalize_from_key(key, gripper_batch)
            .detach()
            .cpu()
            .numpy()
        )
        return unnormed_gripper
    
    # ==================== Action Format Conversion ====================
    def combine_eef_gripper(self, eef_z, gripper_action):
        """
        Combine EEF vectors and gripper into format for IrrepConditionalUnet1D.
        
        IrrepConditionalUnet1D expects: (B, T, 10) = 9D pos/rot + 1D gripper
        
        Args:
            eef_z: (B, T, 3, 3) - 3 vectors (pos, dir1, dir2), each 3D
            gripper_action: (B, T, 1) - gripper state
            
        Returns:
            sample: (B, T, 10) - combined sample for U-net
        """
        B, T, _, _ = eef_z.shape
        # Flatten 3 vectors: (B, T, 3, 3) -> (B, T, 9)
        eef_flat = eef_z.reshape(B, T, 9)
        # Concatenate with gripper: (B, T, 9) + (B, T, 1) -> (B, T, 10)
        sample = torch.cat([eef_flat, gripper_action], dim=-1)
        return sample
    
    def split_eef_gripper(self, sample):
        """
        Split combined sample back into EEF vectors and gripper.
        
        Args:
            sample: (B, T, 10) - combined output from U-net
            
        Returns:
            eef_z: (B, T, 3, 3) - 3 vectors
            gripper: (B, T, 1) - gripper state
        """
        B, T, _ = sample.shape
        eef_flat = sample[..., :9]  # (B, T, 9)
        gripper = sample[..., 9:]   # (B, T, 1)
        eef_z = eef_flat.reshape(B, T, 3, 3)
        return eef_z, gripper
    
    # ==================== Prediction ====================
    def pred_unimanual_traj(self, skill_name_batch, agent_obs, gt_batch=None, task_name_batch=None):
        """
        Predict object-centric trajectory using SDP.
        
        Following reference step.py pattern for inference.
        Uses IrrepConditionalUnet1D for equivariant denoising.
        
        Denoising process (from paper Eq. 2):
        A_t^{k-1} = alpha * (A_t^k - gamma * epsilon_theta(S_t, A_t^k, k) + z)
        """
        pc_data = agent_obs['pc'].repeat(1, self.obs_horizon, 1, 1)
        batch_size = pc_data.shape[0]
        
        ema_nets = self.ema.averaged_model
        
        # Get text conditioning (for fusion into encoder)
        task_skill_condition = self.get_all_embs(skill_name_batch, batch_size, task_name_batch)
        
        # Encode point cloud to spherical features with language fusion
        obs_vec, center, scale = self.proc_pc(pc_data, language_emb=task_skill_condition, ema_nets=ema_nets)
        
        # Prepare global_cond following reference pattern
        # Language is now fused into obs_vec, so no need to concatenate separately
        if self.obs_as_global_cond:
            if "cross_attention" in self.condition_type:
                # obs_vec is already (B, obs_horizon, (c_dim + lang_dim) * irrep_dim)
                global_cond = obs_vec
            else:
                # obs_vec is (B, obs_horizon * (c_dim + lang_dim) * irrep_dim)
                global_cond = obs_vec
        else:
            # Not used as global cond
            global_cond = None
        
        # Initialize noise following reference pattern
        # For IrrepConditionalUnet1D with input_dim = action_dim // 3:
        # - For 3vec: input_dim = 10 // 3 = 3, so sample shape is (B, T, 10) but treated as 3 vectors + 1 scalar
        # - For 4pts: input_dim = 13 // 3 = 4, so sample shape is (B, T, 13) but treated as 4 vectors + 1 scalar
        initial_noise_scale = 1.0
        noisy_sample = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim),
            device=self.device
        ) * initial_noise_scale
        
        # Denoising loop
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        
        curr_sample = noisy_sample
        policy_key = 'unitraj_noise_pred_net'
        
        for k in self.noise_scheduler.timesteps:
            # Predict noise using SDTU
            # IrrepConditionalUnet1D expects: sample (B, T, action_dim), global_cond (B, cond_dim)
            noise_pred = ema_nets[policy_key](
                sample=curr_sample,
                timestep=k,
                global_cond=global_cond,
            )
            
            # Denoise
            curr_sample = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=curr_sample
            ).prev_sample
        
        # Split back into eef and gripper
        if self.eef_representation == "3vec":
            # (B, T, 10) -> (B, T, 9) eef + (B, T, 1) gripper
            pred_eef_flat = curr_sample[..., :9]  # (B, T, 9)
            pred_gripper = curr_sample[..., 9:]   # (B, T, 1)
            pred_eef_z = pred_eef_flat.reshape(batch_size, self.pred_horizon, 3, 3)
        elif self.eef_representation == "4pts":
            # (B, T, 13) -> (B, T, 12) keypoints + (B, T, 1) gripper
            pred_eef_flat = curr_sample[..., :12]  # (B, T, 12)
            pred_gripper = curr_sample[..., 12:]   # (B, T, 1)
            pred_eef_z = pred_eef_flat.reshape(batch_size, self.pred_horizon, 4, 3)
        else:
            raise ValueError(f"Unsupported eef_representation: {self.eef_representation}")
        
        # Recover trajectory
        trans_batch, unnormed_eefpos_xyz, rot6d_batch = self.eef_recover_fn(
            pred_eef_z, scale, center, key="eefpos"
        )
        gripper_batch = self.recover_gripper(pred_gripper, key="gripper")
        
        # Always return predicted trajectories (batch-shaped), and compute metrics if GT is provided.
        action_dict = {
            "eefpos": trans_batch,
            "gripper": gripper_batch,
        }
        eval_metrics = {}

        if gt_batch is not None:
            if self.eef_representation == "4pts":
                gt_4pts = self.eef_proc_fn(gt_batch["eefpos"], "eefpos", center, scale)
                pred_4pts = pred_eef_z
                eval_metrics["pts_error"] = torch.nn.functional.mse_loss(pred_4pts, gt_4pts)

            pred_xyz = trans_batch[:, :, :3, 3]
            gt_xyz = gt_batch["eefpos"][:, :, :3, 3]
            eval_metrics["xyz_l1"] = torch.nn.functional.l1_loss(pred_xyz, gt_xyz)

            gt_Rs = gt_batch["eefpos"][:, :, :3, :3]
            pred_Rs = trans_batch[:, :, :3, :3]
            diff_theta = geodestDist(gt_Rs, pred_Rs).mean()
            eval_metrics["rot_diff"] = diff_theta * 180 / torch.pi

        return action_dict, eval_metrics
    
    # ==================== Forward Pass ====================
    def skill_task_ascii_to_str(self, batch):
        """Convert ASCII tensor batch to string list (following per_skill pattern)."""
        skill_name_batch = ascii_tensor_batch_to_str(batch['skill_name'])
        task_name_batch = ascii_tensor_batch_to_str(batch['task_name'])
        return skill_name_batch, task_name_batch
    
    def forward(self, batch, skill_id=-1):
        """
        Forward pass for evaluation.
        
        Following per_skill_agent pattern.
        """
        batch = to_torch(batch, self.device)
        
        action_dict_all = {}
        eval_metrics_all = {}
        
        assert self.skill_names is not None
        assert self.task_names is not None
        
        skill_name_batch, task_name_batch = self.skill_task_ascii_to_str(batch)
        
        pc_data = batch['pc']
        agent_obs = {'pc': pc_data}
        
        # Use pred_unimanual_traj following per_skill_agent pattern
        action_dict, eval_metrics = self.pred_unimanual_traj(
            skill_name_batch, agent_obs, gt_batch=batch, task_name_batch=task_name_batch
        )
        
        action_dict_all.update(action_dict)
        eval_metrics_all.update(eval_metrics)
        
        denoise_history = []
        return action_dict_all, eval_metrics_all, denoise_history

