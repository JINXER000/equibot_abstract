"""
SDP Encoder wrapper for EquiformerV2
Adapted for object-centric trajectory prediction with per_skill_dataset format.

Reference: paper_ref/sdp/example_paper.tex - Section 3 (Method)
- EquiformerV2 encodes point cloud to spherical Fourier features
- Output: Scene feature C in spherical harmonic space up to degree L
- Achieves SO(3) equivariance via spherical harmonic representations
"""

import math
import torch
import torch.nn as nn
import numpy as np
import einops
from einops import rearrange

from equibot.policies.vision.equiformer_v2.gaussian_rbf import (
    GaussianRadialBasisLayer,
    GaussianRadialBasisLayerFiniteCutoff
)
from equibot.policies.vision.equiformer_v2.edge_rot_mat import init_edge_rot_mat2
from equibot.policies.vision.equiformer_v2.layer_norm import get_normalization_layer
from equibot.policies.vision.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from equibot.policies.vision.equiformer_v2.module_list import ModuleListInfo
from equibot.policies.vision.equiformer_v2.radial_function import RadialFunction
from equibot.policies.vision.equiformer_v2.equiformerv2_block import TransBlock
from equibot.policies.vision.equiformer_v2.connectivity import (
    FpsPool, AdaptiveOriginPool, FpsKnnPool
)


class SDPEncoder(nn.Module):
    """
    SDP Encoder for object-centric trajectory prediction.
    
    Adapts EquiformerV2 to work with per_skill_dataset format:
    - Input: Point cloud [B, T, N, 3 or 6] (object-centric, xyzrgb if color)
    - Output: Spherical Fourier features for diffusion conditioning
    
    Key differences from original EquiFormerEnc:
    - No robot proprioception input (object-centric prediction)
    - Input format adapted for per_skill_dataset
    - Returns both spherical features and canonicalization info (scale, center)
    - Optional language embeddings added with proper irrep structure (type-0 in l=0, m=0 only)
    
    Language embeddings (if provided) are added to s2_feat before flattening,
    following the reference pattern for proprioception features. They are invariant
    scalars (type-0) and placed only in the l=0, m=0 coefficient of the irrep dimension.
    """
    
    def __init__(
        self,
        c_dim=128,
        use_color=False,
        max_neighbors=(16, 16, 16, 16),
        max_radius=(0.05, 0.2, 0.8, 3),
        pool_ratio=(0.25, 0.25, 0.25),
        sphere_channels=(32, 64, 128),
        attn_hidden_channels=(32, 64, 128, 256),
        attn_alpha_channels=(8, 16, 32, 64),
        attn_value_channels=(4, 8, 16, 32),
        ffn_hidden_channels=(32, 64, 128, 256),
        edge_channels=(16, 32, 64, 128),
        num_distance_basis=(64, 64, 64, 64),
        num_heads=4,
        pcd_noise=0,
        norm_type='rms_norm_sh',
        deterministic=False,
        lmax=2,
        mmax=2,
        norm=True,
        grid_resolution=12,
        use_m_share_rad=False,
        distance_function="gaussian_soft",
        use_attn_renorm=True,
        use_grid_mlp=False,
        use_sep_s2_act=True,
        alpha_drop=0.1,
        drop_path_rate=0.,
        proj_drop=0.1,
        weight_init='normal',
        pool_method='fpsknn',
    ):
        super().__init__()
        
        # Store config
        self.c_dim = c_dim
        self.lmax = lmax
        self.mmax = mmax
        self.pcd_noise = pcd_noise
        self.deterministic = deterministic
        self.use_color = use_color
        
        # Sphere channels (last channel is c_dim) 
        assert len(max_neighbors) == len(sphere_channels)
        self.max_neighbors = max_neighbors
        self.pool_ratio = pool_ratio
        self.sphere_channels = sphere_channels + (c_dim,)
        
        self.attn_hidden_channels = attn_hidden_channels 
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        self.lmax_list = [lmax]
        self.mmax_list = [mmax]
        self.grid_resolution = grid_resolution
        self.edge_channels = edge_channels
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis
        self.use_attn_renorm = use_attn_renorm
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop
        self.weight_init = weight_init
        self.max_radius = max_radius
        
        self.n_scales = len(self.max_radius)
        self.num_resolutions = len(self.lmax_list)
        # Number of feature channels passed through the input linear layer.
        # If use_color=False: features are xyz (3).
        # If use_color=True: features are rgb (3), coords remain xyz.
        self.pcd_channels = 3
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels[0]
        
        # Spherical harmonic dimension: (lmax+1)^2
        self.irrep_dim = (lmax + 1) ** 2
        
        # Initialize SO3 rotation modules for Wigner-D matrices
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))
        
        # Coefficient mapping between l and m
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)
        
        # SO3 grid for spherical-grid transformations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m, resolution=self.grid_resolution, normalization='component')
                )
            self.SO3_grid.append(SO3_m_grid)
        
        # Down blocks (hierarchical pooling + transformer)
        self.down_blocks = nn.ModuleList()
        for n in range(len(self.max_neighbors)):
            edge_channels_list = [int(self.num_distance_basis[n])] + [self.edge_channels[n]] * 2
            
            block = nn.ModuleDict()
            
            # Pooling layer
            if n != len(self.max_neighbors) - 1:
                if pool_method == 'fps':
                    block['pool'] = FpsPool(
                        ratio=self.pool_ratio[n],
                        random_start=not self.deterministic,
                        r=self.max_radius[n],
                        max_num_neighbors=self.max_neighbors[n]
                    )
                elif pool_method == 'fpsknn':
                    block['pool'] = FpsKnnPool(
                        ratio=self.pool_ratio[n],
                        random_start=not self.deterministic,
                        r=3,
                        max_num_neighbors=self.max_neighbors[n]
                    )
            else:
                block['pool'] = AdaptiveOriginPool(
                    random_start=not self.deterministic,
                    r=self.max_radius[n],
                    max_num_neighbors=self.max_neighbors[n]
                )
            
            # Distance expansion
            if self.distance_function == 'gaussian':
                block['distance_expansion'] = GaussianRadialBasisLayer(
                    num_basis=self.num_distance_basis[n],
                    cutoff=self.max_radius[n]
                )
            elif self.distance_function == 'gaussian_soft':
                block['distance_expansion'] = GaussianRadialBasisLayerFiniteCutoff(
                    num_basis=self.num_distance_basis[n],
                    cutoff=self.max_radius[n] * 0.99
                )
            
            # Transformer block
            block['transblock'] = TransBlock(
                sphere_channels=self.sphere_channels[n],
                attn_hidden_channels=self.attn_hidden_channels[n],
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels[n],
                attn_value_channels=self.attn_value_channels[n],
                ffn_hidden_channels=self.ffn_hidden_channels[n],
                output_channels=self.sphere_channels[n + 1],
                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,
                SO3_rotation=self.SO3_rotation,
                mappingReduced=self.mappingReduced,
                SO3_grid=self.SO3_grid,
                edge_channels_list=edge_channels_list,
                use_m_share_rad=self.use_m_share_rad,
                use_attn_renorm=self.use_attn_renorm,
                use_grid_mlp=self.use_grid_mlp,
                use_sep_s2_act=self.use_sep_s2_act,
                norm_type=self.norm_type,
                alpha_drop=self.alpha_drop,
                drop_path_rate=self.drop_path_rate,
                proj_drop=self.proj_drop
            )
            
            self.down_blocks.append(block)
        
        # Final normalization
        if norm:
            self.norm = get_normalization_layer(
                self.norm_type,
                lmax=max(self.lmax_list),
                num_channels=self.sphere_channels[-1]
            )
        else:
            self.norm = None
        
        # Input linear layer (project xyz features to sphere channels)
        self.type0_linear = nn.Linear(self.pcd_channels, self.sphere_channels_all, bias=True)
        
        # Weight initialization
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)
        
        print(f"SDPEncoder initialized with {self.num_params} parameters, lmax={lmax}")
    
    def forward(self, pcl, target_norm=1.0, language_emb=None):
        """
        Forward pass for object-centric point cloud encoding.
        
        Args:
            pcl: Point cloud tensor [B, T, N, 3] or [B, T, N, 6]
                - If use_color=False: last dim is xyz.
                - If use_color=True: last dim is xyzrgb (xyz + rgb), mirroring EquiFormerEnc.
            target_norm: Target scale for normalization
            language_emb: Optional language embeddings [B*T, lang_dim] (invariant scalars, type-0)
                - If provided, added to s2_feat with proper irrep structure (l=0, m=0 only)
                - Following reference pattern for proprioception features
            
        Returns:
            dict with:
                - 's2_feat': Spherical Fourier features [B, T, (c_dim + lang_dim) * irrep_dim] if language_emb provided
                           or [B, T, c_dim * irrep_dim] otherwise
                - 'scale': Scale factor [B, T, 1, 1]
                - 'center': Center offset [B, T, 1, 3]
        """
        B, T, N, D = pcl.shape
        assert D in (3, 6), f"Expected point cloud with 3 (xyz) or 6 (xyzrgb) channels, got {D}"
        if self.use_color:
            assert D == 6, f"use_color=True expects pcl shape [B, T, N, 6] (xyzrgb), got {pcl.shape}"
        
        # Reshape for processing: [B*T, N, D]
        pcl_flat = pcl.view(B * T, N, D)
        
        # Split xyz / rgb following EquiFormerEnc when color is present
        xyz = pcl_flat[..., :3]  # [B*T, N, 3]
        rgb = pcl_flat[..., 3:] if self.use_color else None  # [B*T, N, 3] or None
        
        # Compute center and scale for canonicalization on xyz only
        centroid = xyz.mean(dim=1, keepdim=True)  # [B*T, 1, 3]
        xyz_centered = xyz - centroid
        
        z_scale = xyz_centered.norm(dim=-1).mean(dim=-1) / target_norm  # [B*T]
        z_center = centroid  # [B*T, 1, 3]
        
        # Normalize xyz coordinates
        xyz_norm = xyz_centered / (z_scale[:, None, None] + 1e-8)
        
        # Get device and dtype
        device = pcl.device
        dtype = pcl.dtype
        batch_size = B * T
        num_points = N
        
        # Flatten coordinates: [B*T*N, 3]
        node_coord = xyz_norm.view(-1, 3)
        total_points = node_coord.shape[0]
        
        # Node features:
        # - With color: use rgb as features, mirroring EquiFormerEnc.
        # - Without color: node_feature is None (as in reference code).
        if self.use_color:
            node_feature = rgb.view(-1, 3).clone()
        else:
            node_feature = None
        
        # Batch indices
        batch = torch.arange(0, batch_size, device=device).repeat_interleave(num_points)
        
        # Process through down blocks
        node_src = None
        for n, block in enumerate(self.down_blocks):
            # Pooling
            pool_graph = block['pool'](node_coord_src=node_coord, batch_src=batch)
            node_coord_dst, edge_src, edge_dst, degree, batch_dst, node_idx = pool_graph
            
            # Edge vectors
            edge_vec = node_coord.index_select(0, edge_src) - node_coord_dst.index_select(0, edge_dst)
            if not self.deterministic:
                edge_vec = edge_vec + (torch.rand_like(edge_vec) - 0.5) * 1e-6
            edge_vec = edge_vec.detach()
            edge_length = torch.norm(edge_vec, dim=-1).detach()
            
            # Edge rotation matrices
            edge_rot_mat = init_edge_rot_mat2(edge_vec)
            
            # Set Wigner-D matrices
            for i in range(self.num_resolutions):
                self.SO3_rotation[i].set_wigner(edge_rot_mat)
            
            # Initialize source embeddings on first block
            if node_src is None:
                node_src = SO3_Embedding(
                    total_points,
                    self.lmax_list,
                    self.sphere_channels[n],
                    device,
                    dtype,
                )
                
                # Initialize l=0 coefficients from input features
                # If node_feature is None (no color), initialize to zeros
                offset_res = 0
                offset = 0
                for i in range(self.num_resolutions):
                    if node_feature is not None:
                        # Use color features when available
                        if self.num_resolutions == 1:
                            node_src.embedding[:, offset_res, :] = self.type0_linear(node_feature)
                        else:
                            node_src.embedding[:, offset_res, :] = self.type0_linear(node_feature)[
                                :, offset:offset + self.sphere_channels[0]
                            ]
                    else:
                        # No features: initialize to zeros (as in reference when no color)
                        node_src.embedding[:, offset_res, :] = 0.0
                    offset = offset + self.sphere_channels[0]
                    offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)
            
            # Distance expansion
            edge_attr = block['distance_expansion'](edge_length)
            
            # Create destination embedding
            node_dst = SO3_Embedding(
                batch_size,
                self.lmax_list,
                self.sphere_channels[n],
                device,
                dtype,
            )
            
            if n != len(self.max_neighbors) - 1:
                node_dst.set_embedding(node_src.embedding[node_idx])
            node_dst.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())
            
            # Transformer block
            node_dst = block['transblock'](
                node_src, node_dst, edge_attr, edge_src, edge_dst, batch=batch
            )
            
            node_src = node_dst
            node_coord = node_coord_dst
            batch = batch_dst
        
        # Final normalization
        if self.norm is not None:
            node_dst.embedding = self.norm(node_dst.embedding)
        
        # Output spherical features: [B*T, irrep_dim, c_dim]
        s2_feat = node_dst.embedding
        
        # Add language embeddings if provided (following reference pattern for proprioception)
        # Language embeddings are invariant scalars (type-0), placed only in l=0, m=0
        if language_emb is not None:
            lang_dim = language_emb.shape[-1]  # [B*T, lang_dim]
            # Create language feature tensor with proper irrep structure
            lang_feat = torch.zeros(batch_size, self.irrep_dim, lang_dim, device=device, dtype=dtype)
            # Place language embeddings only in l=0, m=0 (index 0)
            lang_feat[:, 0, :] = language_emb  # [B*T, lang_dim] -> [B*T, 1, lang_dim] placed at irrep index 0
            # Concatenate along channel dimension: [B*T, irrep_dim, c_dim + lang_dim]
            s2_feat = torch.cat([s2_feat, lang_feat], dim=-1)
        
        # Flatten: [B*T, irrep_dim, c_dim] or [B*T, irrep_dim, c_dim + lang_dim] -> [B*T, (c_dim + lang_dim) * irrep_dim]
        s2_feat = einops.rearrange(s2_feat, 'bt irrep c -> bt (c irrep)')
        
        # Reshape outputs to [B, T, ...]
        s2_feat = s2_feat.view(B, T, -1)
        z_scale = z_scale.view(B, T, 1, 1)
        z_center = z_center.view(B, T, 1, 3)
        
        return {
            's2_feat': s2_feat,  # [B, T, c_dim * irrep_dim]
            'scale': z_scale,    # [B, T, 1, 1]
            'center': z_center,  # [B, T, 1, 3]
        }
    
    def output_dim(self, language_dim=0):
        """
        Return output feature dimension (channel dimension, before irrep flattening).
        This matches the reference equiformer_enc.py output_shape() convention.
        
        Args:
            language_dim: Dimension of language embeddings (default 0)
        
        Returns:
            Channel dimension: c_dim + language_dim
        """
        return self.c_dim + language_dim
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, SO3_LinearV2)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                nn.init.normal_(m.weight, 0, std)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)
    
    def _uniform_init_linear_weights(self, m):
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            nn.init.uniform_(m.weight, -std, std)

