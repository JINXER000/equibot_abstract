import numpy as np
import torch
import torch.nn.functional as F
import pathlib
import numpy as np
import json
import networkx as nx

EQUIBOT_PATH = pathlib.Path(__file__).parent.parent.parent.parent.absolute()

def to_torch(batch, device):    return {k: v.to(device) for k, v in batch.items()}

def collate_fn(batch):
    """
    Custom collate function to handle variable-length skill name tensors.
    Pads skill_name tensors to the same length for batching.
    """
    if "skill_name" not in batch[0]:
        return torch.utils.data.dataloader.default_collate(batch)
    
    # Ensure both name tensors are 1-D integer tensors with a consistent dtype
    for item in batch:
        if not torch.is_tensor(item['skill_name']):
            item['skill_name'] = torch.tensor(item['skill_name'], dtype=torch.long)
        else:
            item['skill_name'] = item['skill_name'].to(dtype=torch.long)
        if not torch.is_tensor(item['task_name']):
            item['task_name'] = torch.tensor(item['task_name'], dtype=torch.long)
        else:
            item['task_name'] = item['task_name'].to(dtype=torch.long)
    
    # Find the maximum length of skill_name/task_name tensors in the batch
    max_skill_name_len = max(len(item['skill_name']) for item in batch)
    max_task_name_len = max(len(item['task_name']) for item in batch)
    
    # Pad all skill_name/task_name tensors to the same length
    for item in batch:
        skill_name_len = len(item['skill_name'])
        if skill_name_len < max_skill_name_len:
            padding = torch.zeros(max_skill_name_len - skill_name_len, dtype=torch.long)
            item['skill_name'] = torch.cat([item['skill_name'], padding])

        task_name_len = len(item['task_name'])
        if task_name_len < max_task_name_len:
            padding = torch.zeros(max_task_name_len - task_name_len, dtype=torch.long)
            item['task_name'] = torch.cat([item['task_name'], padding])
    
    # Use default collate for the rest
    return torch.utils.data.dataloader.default_collate(batch)

def to_tensor(obs):
    return {k: torch.tensor(v).float() for k, v in obs.items()}

def to_np(obs):
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            obs[k] = v.cpu().detach().numpy()
        else:
            obs[k] = np.array(v)
    return obs

def rotate_around_z(
    points,
    angle_rad=0.0,
    center=np.array([0.0, 0.0, 0.0]),
    scale=np.array([1.0, 1.0, 1.0]),
):
    """
    Rotate points around Z axis.
    
    Args:
        points: Point cloud of shape (N, D) or (D,) where D >= 3.
            First 3 channels are xyz (rotated).
            Additional channels (e.g., rgb at channels 3:6) are preserved unchanged.
        angle_rad: Rotation angle in radians.
        center: Center of rotation (3,).
        scale: Scale factor (3,).
        
    Returns:
        Rotated points with same shape as input.
    """
    # Check if the input points have the correct shape (at least 3 channels for xyz)
    assert (len(points.shape) == 1 and len(points) >= 3) or points.shape[-1] ==6
    p_shape = points.shape
    num_channels = p_shape[-1]
    
    # Reshape to (N, D)
    points_flat = points.reshape(-1, num_channels)
    
    # Extract xyz and extra channels (e.g., rgb)
    xyz = points_flat[:, :3] - center[None]
    extra_channels = points_flat[:, 3:] if num_channels > 3 else None

    # Create the rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
    )

    # Apply the rotation to xyz only
    rotated_xyz = np.dot(xyz, rotation_matrix.T) * scale[None] + center[None]
    
    # Combine rotated xyz with preserved extra channels (rgb)
    if extra_channels is not None:
        rotated_points = np.concatenate([rotated_xyz, extra_channels], axis=-1)
    else:
        rotated_points = rotated_xyz
    
    rotated_points = rotated_points.reshape(p_shape)

    return rotated_points

### The function is to test the equivariance of the model
## input np or torch tensor, output np
def rotate_observation(np_obs, yaw_rotation):

    from equibot.envs.sim_mobile.utils.transformations import euler2mat
    rot_3x3 = euler2mat([0, 0, yaw_rotation]) 
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rot_3x3

    obs_rotated = np_obs.copy()
    for k, v in np_obs.items():
        if k.endswith('pc'):
            pc_np = v
            rotated_pc = rotate_around_z(pc_np, yaw_rotation)
            obs_rotated[k] = rotated_pc

        elif k.endswith('grasp')  or k.endswith('eefpos'):
            grasp_np = v

            assert len(grasp_np.shape) == 4  # B, 1, 8, 4
            assert trans_mat.shape == (4, 4)  # Transformation matrix should be 4x4

            # Extract pre-grasp and eff-grasp components
            pre_grasp = grasp_np[:, :, :4, :]  # Extract first 4 rows along the second last axis
            rotated_grasp = np.einsum('ij,bnjk->bnik', trans_mat, pre_grasp)  # Batched matrix multiplication

            if grasp_np.shape[2] == 8:
                eff_grasp = grasp_np[:, :, 4:, :]  # Extract last 4 rows along the second last axis
                rotated_eff_grasp = np.einsum('ij,bnjk->bnik', trans_mat, eff_grasp)  # Batched matrix multiplication
                # Combine back along the third axis
                rotated_grasp = np.concatenate([rotated_grasp, rotated_eff_grasp], axis=2)

            obs_rotated[k] = rotated_grasp

    return obs_rotated


def get_env_class(env_name):
    if env_name == "fold":
        from equibot.envs.sim_mobile.folding_env import FoldingEnv
        return FoldingEnv
    elif env_name == "cover":
        from equibot.envs.sim_mobile.covering_env import CoveringEnv
        return CoveringEnv
    elif env_name == "close":
        from equibot.envs.sim_mobile.closing_env import ClosingEnv
        return ClosingEnv
    elif env_name == "insert":
        from equibot.envs.sim_mobile.insertion_env_todo import InsertionEnv
        return InsertionEnv
    else:
        raise ValueError()

def get_dataset(cfg, mode="train"):
    if 'dataset_type' not in cfg.data.dataset:
        from equibot.policies.datasets.dataset import BaseDataset
        return BaseDataset(cfg.data.dataset, mode)
    dataset_type = cfg.data.dataset.dataset_type
    if dataset_type == "hdf5_mini":
        from equibot.policies.datasets.abstract_dataset import ALOHAPoseDataset
        return ALOHAPoseDataset(cfg.data.dataset, mode)
    elif dataset_type == "dual_hdf5_mini":
        from equibot.policies.datasets.dual_abs_dataset import DualAbsDataset
        return DualAbsDataset(cfg.data.dataset, mode)
    elif dataset_type == "mj_insertion_pred":
        from equibot.policies.datasets.dual_abs_dataset import DualAbsDataset
        return DualAbsDataset(cfg.data.dataset, mode)
    elif "robosuite" in dataset_type:
        from equibot.policies.datasets.dmg_dataset import RobosuiteDataset
        return RobosuiteDataset(cfg.data.dataset, mode)
    elif dataset_type == "dmg_policy":
        from equibot.policies.datasets.robosuite_policy_dataset import RobosuitePolicyDataset
        return RobosuitePolicyDataset(cfg.data.dataset, mode)
    elif "per_skill" in dataset_type:
        from equibot.policies.datasets.per_skill_dataset import PerSkillDataset
        return PerSkillDataset(cfg.data.dataset, mode)
    elif dataset_type == "real_aloha_traj":
        from equibot.policies.datasets.real_aloha_dataset import RealAlohaDataset
        return RealAlohaDataset(cfg.data.dataset, mode)
    else:
        raise ValueError(f"Dataset type [{dataset_type}] not supported.")



def get_agent(agent_name):
    if agent_name == "dp":
        from equibot.policies.agents.dp_agent import DPAgent
        return DPAgent
    elif agent_name == "equibot":
        from equibot.policies.agents.equibot_agent import EquiBotAgent
        return EquiBotAgent
    elif agent_name == "aloha":
        from equibot.policies.agents.aloha_agent import ALOHAAgent
        return ALOHAAgent
    elif agent_name == "compaloha":
        from equibot.policies.agents.compaloha_agent import CompALOHAAgent
        return CompALOHAAgent
    elif agent_name == "traj":
        from equibot.policies.agents.traj_agent import TrajAgent
        return TrajAgent
    elif agent_name == "dmg":
        from equibot.policies.agents.dmg_agent import DMGAgent
        return DMGAgent
    elif agent_name == "eefequibot":
        from equibot.policies.agents.eefequibot_agent import EefEquiBotAgent
        return EefEquiBotAgent
    elif agent_name == "per_skill":
        from equibot.policies.agents.per_skill_agent import EquiSkillAgent
        return EquiSkillAgent
    elif agent_name == "sdp":
        from equibot.policies.agents.sdp_agent import SDPAgent
        return SDPAgent
    else:
        raise ValueError(f"Agent with name [{agent_name}] not found.")

def get_agent_from_ckpt(ckpt_path):
    state_dict = torch.load(ckpt_path)
    cfg = state_dict["cfg"]
    agent_name = cfg.agent.agent_name
    agent = get_agent(agent_name)(cfg)
    agent.load_state_dict_to_actor(state_dict)
    return agent

# impl from: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#rotation_6d_to_matrix
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    # a1, a2 = d6[..., :3], d6[..., 3:]
    # b1 = F.normalize(a1, dim=-1)
    # b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    # b2 = F.normalize(b2, dim=-1)
    # b3 = torch.cross(b1, b2, dim=-1)
    # return torch.stack((b1, b2, b3), dim=-2)

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2).transpose(-1, -2)  

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # batch_dim = matrix.size()[:-2]
    # return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

    batch_dim = matrix.size()[:-2]
    transpose_matrix = matrix.transpose(-1, -2)
    return transpose_matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def geodestDist(Rgts, Rps):
    # Compute Rgts^T @ Rps in batch
    Rds = torch.matmul(Rgts.transpose(-1, -2), Rps)
    
    # Compute trace for each rotation matrix in batch
    Rt = Rds.diagonal(dim1=-2, dim2=-1).sum(-1)
    
    # Clamp for numerical stability and compute theta
    theta = torch.acos(torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6))
    
    return theta

## gripper_pcd is (B, H, 4, 3)
def compute_plane_normal(gripper_pcd):
    # Check if inputs are tensors or numpy arrays
    is_tensor = torch.is_tensor(gripper_pcd)
    
    if is_tensor:
        # Tensor operations - apply to last 2 dims (4, 3)
        x1 = gripper_pcd[..., 0, :]  # Shape: (B, H, 3)
        x2 = gripper_pcd[..., 1, :]  # Shape: (B, H, 3)
        x4 = gripper_pcd[..., 3, :]  # Shape: (B, H, 3)
        v1 = x2 - x1  # Shape: (B, H, 3)
        v2 = x4 - x1  # Shape: (B, H, 3)
        normal = torch.cross(v1, v2, dim=-1)  # Shape: (B, H, 3)
        norm = torch.norm(normal, dim=-1, keepdim=True)  # Shape: (B, H, 1)
        return normal / (norm + 1e-8)  # Add small epsilon for numerical stability
    else:
        # NumPy operations
        x1 = gripper_pcd[..., 0, :]
        x2 = gripper_pcd[..., 1, :]
        x4 = gripper_pcd[..., 3, :]
        v1 = x2 - x1
        v2 = x4 - x1
        normal = np.cross(v1, v2)
        return normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)

## input:  (B, H, 3),  (B, H, 3)
def rotation_matrix_from_vectors(v1, v2):
    """
    Find the rotation matrix that aligns v1 to v2
    :param v1: A 3d "source" vector (B, H, 3) or (B, H, 1, 3)
    :param v2: A 3d "destination" vector (B, H, 3) or (B, H, 1, 3)
    :return mat: A transform matrix (B, H, 3, 3) which when applied to v1, aligns it with v2.
    """
    # Check if inputs are tensors or numpy arrays
    is_tensor = torch.is_tensor(v1)
    
    if is_tensor:
        assert len(v1.shape) == 3
        assert len(v2.shape) == 3
        # Tensor operations
        v1_norm = torch.norm(v1, dim=-1, keepdim=True)
        v2_norm = torch.norm(v2, dim=-1, keepdim=True)
        v1 = v1 / (v1_norm + 1e-8)
        v2 = v2 / (v2_norm + 1e-8)
        
        axis = torch.cross(v1, v2, dim=-1)  # Shape: (B, H, 3)
        axis_len = torch.norm(axis, dim=-1, keepdim=True)  # Shape: (B, H, 1)
        
        # Handle zero axis case
        axis = torch.where(axis_len > 1e-8, axis / (axis_len + 1e-8), torch.zeros_like(axis))
        
        # Compute angle
        cos_angle = torch.sum(v1 * v2, dim=-1, keepdim=True)  # Shape: (B, H, 1)
        cos_angle = torch.clamp(cos_angle, -1, 1)
        angle = torch.acos(cos_angle)  # Shape: (B, H, 1)
        
        # Create skew-symmetric matrix K for each batch element
        # K = [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        batch_shape = v1.shape[:-1]  # (B, H)
        K = torch.zeros(*batch_shape, 3, 3, device=v1.device, dtype=v1.dtype)
        
        # Fill K matrix
        K[..., 0, 1] = -axis[..., 2]
        K[..., 0, 2] = axis[..., 1]
        K[..., 1, 0] = axis[..., 2]
        K[..., 1, 2] = -axis[..., 0]
        K[..., 2, 0] = -axis[..., 1]
        K[..., 2, 1] = axis[..., 0]
        
        # Compute rotation matrix using Rodrigues' formula
        I = torch.eye(3, device=v1.device, dtype=v1.dtype).expand(*batch_shape, 3, 3)
        sin_angle = torch.sin(angle).unsqueeze(-1)  # Shape: (B, H, 1, 1)
        cos_angle_term = (1 - torch.cos(angle)).unsqueeze(-1)  # Shape: (B, H, 1, 1)
        
        R = I + sin_angle * K + cos_angle_term * torch.matmul(K, K)
        return R
    else:
        # NumPy operations (original implementation)
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        axis = np.cross(v1, v2)
        axis_len = np.linalg.norm(axis)
        if axis_len != 0:
            axis = axis / axis_len
        angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))

        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R

def convert_trans_to_4pts(grasp_trans_arr, original_gripper_pcd):
    # Check if inputs are tensors or numpy arrays
    is_tensor = torch.is_tensor(grasp_trans_arr)
    
    if is_tensor:
        # Tensor operations
        batch_size, horizon, _, _ = grasp_trans_arr.shape
        grasp_xyz = grasp_trans_arr[:, :, :3, 3].reshape(batch_size, horizon, 1, 3)  # B, H, 1, 3
        grasp_rot = grasp_trans_arr[:, :, :3, :3].reshape(-1, 3, 3)  # B*H, 3, 3
        
        # Convert original_gripper_pcd to tensor if it's not already
        if not torch.is_tensor(original_gripper_pcd):
            original_gripper_pcd = torch.tensor(original_gripper_pcd, device=grasp_trans_arr.device, dtype=grasp_trans_arr.dtype)
        
        ## expand to B, H, 4, 3
        original_gripper_pcd = original_gripper_pcd.reshape(1,1,4,3).repeat(batch_size, horizon, 1, 1)

        ## canonicalize the gripper pcd, [3] should be origin
        original_pcd = original_gripper_pcd - original_gripper_pcd[:, :, 3:, :]  
        original_pcd_batch = original_pcd.reshape(batch_size * horizon, 4, 3)
        ## do rotation in the last two dims
        rotated_pcd = torch.matmul(original_pcd_batch, grasp_rot.transpose(-2, -1))  # (B*H, 4, 3)
        ## reshape back to (B, H, 4, 3) and add translation
        rotated_pcd = rotated_pcd.reshape(batch_size, horizon, -1, 3)
        gripper_pcd = rotated_pcd + grasp_xyz
    else:
        # NumPy operations (original implementation)
        batch_size, horizon, _, _ = grasp_trans_arr.shape
        grasp_xyz = grasp_trans_arr[:, :, :3, 3].reshape(batch_size, horizon, 1, 3)  # B, H, 1, 3
        grasp_rot = grasp_trans_arr[:, :, :3, :3].reshape(-1, 3, 3)  # B*H, 3, 3
        original_pcd = original_gripper_pcd - original_gripper_pcd[3]
        rotated_pcd = np.dot(original_pcd, grasp_rot.T)
        gripper_pcd = rotated_pcd + grasp_xyz
        gripper_pcd = gripper_pcd.reshape(batch_size, horizon, -1, 3)
    
    return gripper_pcd

def convert_4pts_to_trans(gripper_pcd, origin_gripper_pcd):
    # Check if inputs are tensors or numpy arrays
    is_tensor = torch.is_tensor(gripper_pcd)
    
    if is_tensor:
        # Tensor operations
        batch_size, horizon, _, _ = gripper_pcd.shape
        
        # Convert origin_gripper_pcd to tensor if needed
        if not torch.is_tensor(origin_gripper_pcd):
            origin_gripper_pcd = torch.tensor(origin_gripper_pcd, device=gripper_pcd.device, dtype=gripper_pcd.dtype)

        ## expand to B, H, 4, 3
        origin_gripper_pcd = origin_gripper_pcd.reshape(1,1,4,3).repeat(batch_size, horizon, 1, 1)
        
        # Compute plane normals - shape: (B, H, 3)
        origin_plane_normal = compute_plane_normal(origin_gripper_pcd)
        pred_plane_normal = compute_plane_normal(gripper_pcd)
        # assert len(origin_plane_normal.shape) == 3
        # assert len(pred_plane_normal.shape) == 3
        
        # Get rotation matrix from plane normals - shape: (B, H, 3, 3)
        plane_rotation = rotation_matrix_from_vectors(origin_plane_normal, pred_plane_normal)
        
        # Compute reference vectors - shape: (B, H, 3)
        origin_ref_vector = origin_gripper_pcd[..., 3:, :] - origin_gripper_pcd[..., :1, :]
        pred_ref_vector = gripper_pcd[..., 3:, :] - gripper_pcd[..., :1, :]
        
        # Apply plane rotation to origin reference vector
        # origin_ref_vector_rotated = torch.matmul(plane_rotation, origin_ref_vector)
        origin_ref_vector_rotated = torch.matmul(origin_ref_vector, plane_rotation.transpose(-2, -1))
        
        # Get in-plane rotation. Note, we have to make sure the input size to be B, H, 3
        origin_ref_vector_rotated = origin_ref_vector_rotated.reshape(batch_size, horizon, 3)
        pred_ref_vector = pred_ref_vector.reshape(batch_size, horizon, 3)
        in_plane_rotation = rotation_matrix_from_vectors(origin_ref_vector_rotated, pred_ref_vector)
        
        # Combine rotations
        full_rotation = torch.matmul(in_plane_rotation, plane_rotation)
        
        # Get gripper position - shape: (B, H, 3)
        gripper_pos = gripper_pcd[..., 3, :]
        
        # Create transformation matrix
        grasp_trans_arr = torch.zeros((batch_size, horizon, 4, 4), device=gripper_pcd.device, dtype=gripper_pcd.dtype)
        grasp_trans_arr[..., :3, :3] = full_rotation
        grasp_trans_arr[..., :3, 3] = gripper_pos
        grasp_trans_arr[..., 3, 3] = 1
        
        return grasp_trans_arr
    else:
        # NumPy operations
        batch_size, horizon, _, _ = gripper_pcd.shape
        
        # Convert to numpy if needed
        if torch.is_tensor(origin_gripper_pcd):
            origin_gripper_pcd = origin_gripper_pcd.detach().cpu().numpy()
        if torch.is_tensor(gripper_pcd):
            gripper_pcd = gripper_pcd.detach().cpu().numpy()
        
        # Compute plane normals
        origin_plane_normal = compute_plane_normal(origin_gripper_pcd)
        pred_plane_normal = compute_plane_normal(gripper_pcd)
        
        # Get rotation matrix from plane normals
        plane_rotation = rotation_matrix_from_vectors(origin_plane_normal, pred_plane_normal)
        
        # Compute reference vectors
        origin_ref_vector = origin_gripper_pcd[..., 3, :] - origin_gripper_pcd[..., 0, :]
        pred_ref_vector = gripper_pcd[..., 3, :] - gripper_pcd[..., 0, :]
        
        # Apply plane rotation to origin reference vector
        origin_ref_vector_rotated = np.matmul(plane_rotation, origin_ref_vector[..., np.newaxis]).squeeze(-1)
        
        # Get in-plane rotation
        in_plane_rotation = rotation_matrix_from_vectors(origin_ref_vector_rotated, pred_ref_vector)
        
        # Combine rotations
        full_rotation = np.matmul(in_plane_rotation, plane_rotation)
        
        # Get gripper position
        gripper_pos = gripper_pcd[..., 3, :]
        
        # Create transformation matrix
        grasp_trans_arr = np.zeros((batch_size, horizon, 4, 4))
        grasp_trans_arr[..., :3, :3] = full_rotation
        grasp_trans_arr[..., :3, 3] = gripper_pos
        grasp_trans_arr[..., 3, 3] = 1
        
        return grasp_trans_arr

def convert_trans_to_vec(grasp_trans_arr, has_eff=False):
    batch_size, horizon, _, _ = grasp_trans_arr.shape
    grasp_xyz = grasp_trans_arr[:, :, :3, 3].reshape(batch_size, horizon, 1, 3)  # B, H, 1, 3
    grasp_rot =  grasp_trans_arr[:, :, :3, :3].reshape(-1, 3, 3) # B*H, 3, 3
    rot6d = matrix_to_rotation_6d(grasp_rot) # B*H, 6
    rot_dir1 = rot6d[:, :3].reshape(batch_size, horizon, 1, 3)
    rot_dir2 = rot6d[:, 3:].reshape(batch_size, horizon, 1, 3)

    if has_eff:
        eff_grasp_xyz = grasp_trans_arr[:, :, 4:7, 3].reshape(batch_size, horizon, 1, 3)
        eff_grasp_rot =  grasp_trans_arr[:, :, 4:7, :3].reshape(-1, 3, 3)
        eff_rot6d = matrix_to_rotation_6d(eff_grasp_rot)
        eff_rot_dir1 = eff_rot6d[:, :3].reshape(batch_size, horizon, 1, 3)
        eff_rot_dir2 = eff_rot6d[:, 3:].reshape(batch_size, horizon, 1, 3)

        # combine pred and eff tensors
        grasp_xyz = torch.cat((grasp_xyz, eff_grasp_xyz), dim=2)
        rot_dir1 = torch.cat((rot_dir1, eff_rot_dir1), dim=2)
        rot_dir2 = torch.cat((rot_dir2, eff_rot_dir2), dim=2)

    return grasp_xyz, rot_dir1, rot_dir2

def convert_vec_to_trans(rot6d_batch, unnormed_grasp_xyz, has_eff = False):
    batch_size, horizon, grasp_num, vec_dim = rot6d_batch.shape
    if has_eff == False:
        assert grasp_num == 1
        rot6d_batch = rot6d_batch.reshape(-1, 6)
        rotation_mat_ts = rotation_6d_to_matrix(rot6d_batch)
        rotation_mat = rotation_mat_ts

        trans_mat_batch = torch.zeros((batch_size * horizon, 4, 4), device=rot6d_batch.device)
        trans_mat_batch[:, :3, :3] = rotation_mat
        trans_mat_batch[:, :3, 3] = unnormed_grasp_xyz.reshape(-1, 3)
        trans_mat_batch[:, 3, 3] = 1

        trans_mat_batch = trans_mat_batch.reshape(batch_size, horizon, 4, 4)
    else:
        assert grasp_num == 2
        unnormed_grasp_xyz_pre = unnormed_grasp_xyz[:, :, 0, :].reshape(-1, 3)
        rot6d_batch_pre = rot6d_batch[:, :, 0, :].reshape(-1, 6)
        rotation_mat_pre = rotation_6d_to_matrix(rot6d_batch_pre)
        unnormed_grasp_xyz_eff = unnormed_grasp_xyz[:, :, 1, :].reshape(-1, 3)
        rot6d_batch_eff = rot6d_batch[:, :, 1, :].reshape(-1, 6)
        rotation_mat_eff = rotation_6d_to_matrix(rot6d_batch_eff)

        trans_mat_batch = torch.zeros((batch_size * horizon, 8, 4), device=rot6d_batch.device)
        trans_mat_batch[:, :3, :3] = rotation_mat_pre
        trans_mat_batch[:, :3, 3] = unnormed_grasp_xyz_pre.reshape(-1, 3)
        trans_mat_batch[:, 3, 3] = 1
        trans_mat_batch[:, 4:7, :3] = rotation_mat_eff
        trans_mat_batch[:, 4:7, 3] = unnormed_grasp_xyz_eff.reshape(-1, 3)
        trans_mat_batch[:, 7, 3] = 1

    trans_mat_batch = trans_mat_batch.reshape(batch_size, horizon, -1, 4)
    return trans_mat_batch

def rotate_vec_grasp(grasp, rot_z):
    ## vectorize the  grasp
    pred_grasp_trans = grasp[:, :4, :].reshape(1, 1, 4, 4)
    # pred_grasp_trans[:, :, :3, :3] = pred_grasp_trans[:, :, :3, :3].transpose(-2, -1)
    grasp_xyz, grasp_dir1, grasp_dir2 = convert_trans_to_vec(pred_grasp_trans, has_eff = False)
    gt_grasp_z = torch.cat([grasp_xyz, grasp_dir1, grasp_dir2], dim=-2)

    gt_z_np = gt_grasp_z.detach().cpu().numpy()
    rotated_gt_z = rotate_around_z(gt_z_np, rot_z)
    rotated_grasp_vec = torch.tensor(rotated_gt_z).float()

    # rotated_grasp_vec = torch.einsum('bnij, ', gt_grasp_z, torch.tensor(rotation_matrix).float())
    rotated_rot6d = rotated_grasp_vec[:, :,  1:, :].reshape(-1, 1, 1, 6)
    rotated_xyz = rotated_grasp_vec[:, :, 0, :].reshape(-1, 1, 1, 3)
    rotated_grasp_trans = convert_vec_to_trans(rotated_rot6d, rotated_xyz, has_eff = False)
    # rotated_grasp_trans[:, :, :3, :3] = rotated_grasp_trans[:, :, :3, :3].transpose(-2, -1)

    return rotated_grasp_trans

class ActionSlice(object):
    def __init__(self, mode = "separated"):
        self.mode = mode
        if mode == "separated":
            self.data = {'left_jpose': None, 'right_jpose': None, \
                         'left_grasp': None, 'right_grasp': None}
            self.ee_dof = 7
        elif mode == "combined":
            self.data = {'jpose': None, 'grasp': None}
            self.ee_dof = 14

    def update(self, key, value):
        if key == 'dual_jpose':
            self.data['left_jpose'] = value[:self.ee_dof]
            self.data['right_jpose'] = value[self.ee_dof:]
        else:
            self.data[key] = value

    def get(self, key):
        if not key in self.data:
            return None
        
        return self.data[key]
    

def anneal_loss_scaling( vec_loss, scalar_loss, current_epoch, total_epochs):
    alpha = min(1, current_epoch / total_epochs)
    ## bias to vector loss at the begining, but gradually shift to scalar loss
    loss = (1 - alpha) * vec_loss + (alpha) * scalar_loss
    return loss

def origin_loss_scaling(vec_loss, scalar_loss):
    n_vec = np.prod(vec_loss.shape)
    n_scalar = np.prod(scalar_loss.shape)
    k = n_vec / (n_vec + n_scalar)
    loss = k * vec_loss + (1 - k) * scalar_loss
    return loss

def manual_loss_scaling(vec_loss, scalar_loss, alpha):
    loss = alpha * vec_loss + (1 - alpha) * scalar_loss
    return loss


### for dmg agent
def str_to_ascii_tensor(text: str) -> torch.Tensor:
    """将字符串转换为 ASCII 值的 torch.Tensor"""
    ascii_values = [ord(char) for char in text]  # 获取每个字符的 ASCII 值
    return torch.tensor(ascii_values, dtype=torch.int32)  # 使用 int32 存储

def ascii_tensor_to_str(tensor: torch.Tensor) -> str:
    """将 ASCII 值的 Tensor 还原为字符串"""
    if tensor.dim() == 0:  # 处理单个数字（标量）的情况
        return chr(int(tensor.item()))
    
    # Remove padding (zeros) before converting to string
    # Find the first zero or end of tensor
    tensor_list = tensor.tolist()
    # Find the first zero (padding) or use the full length
    end_idx = len(tensor_list)
    for i, val in enumerate(tensor_list):
        if val == 0:  # Padding value
            end_idx = i
            break
    
    # Convert only the non-padded part to string
    return ''.join([chr(int(code)) for code in tensor_list[:end_idx]])

def ascii_tensor_batch_to_str(tensor_batch: torch.Tensor) -> list:
    """将 ASCII 值的 Tensor batch 还原为字符串列表"""
    if tensor_batch.dim() == 1:  # 单个样本的情况
        return [ascii_tensor_to_str(tensor_batch)]
    elif tensor_batch.dim() == 2:  # 批量样本的情况
        return [ascii_tensor_to_str(tensor) for tensor in tensor_batch]
    else:
        raise ValueError(f"Unsupported tensor batch dimension: {tensor_batch.dim()}")

def get_skill_names(cpu_obs):
    data_keys = list(cpu_obs.keys())
    skill_names = []
    for key in data_keys:
        skill_name = key.split(':')[0]
        skill_names.append(skill_name)
    return set(skill_names)

from scipy.spatial.transform import Rotation

## quaternion is (x, y, z, w)
def compose_transformation(xyz, quat):
    rot_mat = Rotation.from_quat(quat).as_matrix()
    trans = np.concatenate([np.concatenate([rot_mat, np.array([xyz]).T], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
    return trans



import open3d as o3d

def downsample_pc(pc, num_points, method = 'random', debug_visualize = False):
    """
    Downsample point cloud to num_points.
    
    Args:
        pc: Point cloud array of shape (N, D) where D >= 3.
            First 3 channels are xyz (used for geometry-based selection).
            Additional channels (e.g., rgb) are preserved.
        num_points: Target number of points.
        method: 'random', 'fps', or 'uniform'.
        debug_visualize: If True, save downsampled point cloud to file.
        
    Returns:
        Downsampled point cloud of shape (num_points, D).
    """
    use_pc_color = pc.shape[1] > 3

    if pc.shape[0] < num_points * 0.3:
        raise ValueError('Input pc shape is not enough points!')
    elif pc.shape[0] < num_points:
        random_repeated_indices = np.random.choice(pc.shape[0], num_points - pc.shape[0], replace=True)
        pc = np.concatenate([pc, pc[random_repeated_indices]], axis=0)
        return pc
    elif pc.shape[0] == num_points:
        return pc

    pcd = o3d.geometry.PointCloud()
    

    # Extract xyz for geometry-based downsampling
    xyz = pc[:, :3]
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if use_pc_color:
        rgb = pc[:, 3:]
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        rgb = None
    
    if method == 'random':
        selected_ids = np.random.choice(pc.shape[0], num_points, replace=False)
        pcd_down = pcd.select_by_index(selected_ids)
    elif method == 'fps':
        pcd_down = pcd.farthest_point_down_sample(num_points)
    elif method == 'uniform':
        every_k_points = max(1, pc.shape[0] // num_points)
        pcd_down = pcd.uniform_down_sample(every_k_points=every_k_points)
    else:
        raise ValueError(f'Method {method} not supported!')
    
    down_pts = np.asarray(pcd_down.points)
    if use_pc_color:
        down_pts_rgb = np.asarray(pcd_down.colors)
        down_pts = np.concatenate([down_pts, down_pts_rgb], axis=-1)
    # Ensure we have exactly num_points
    if down_pts.shape[0] < num_points:
        random_repeated_indices = np.random.choice(down_pts.shape[0], num_points - down_pts.shape[0], replace=True)
        down_pts = np.concatenate([down_pts, down_pts[random_repeated_indices]], axis=0)

    ## save the pc
    if debug_visualize:
        pcd_debug = o3d.geometry.PointCloud()
        pcd_debug.points = o3d.utility.Vector3dVector(down_pts[:, :3])
        if use_pc_color:
            pcd_debug.colors = o3d.utility.Vector3dVector(down_pts[:, 3:])
        o3d.io.write_point_cloud(f'{method}_pc.ply', pcd_debug)

    return down_pts

def add_projected_point(pc, num_ratio = 0.5):
    """
    Project the point cloud onto the plane at the minimum z value, then downsample.
    
    Args:
        pc (np.ndarray): Input point cloud of shape (N, D) where D >= 3.
            First 3 channels are xyz. Additional channels (e.g., rgb) are preserved
            using average values for projected points.
    Returns:
        np.ndarray: Downsampled projected point cloud of shape (num_points, D)
    """
    # Find the minimum z value
    min_z = np.min(pc[:, 2])
    # Project all points onto the plane z = min_z
    projected_pc = pc.copy()
    projected_pc[:, 2] = min_z
    
    num_points = int(pc.shape[0] * num_ratio)
    projected_pc_down = downsample_pc(projected_pc, num_points, method = 'random')
    
    # If there are additional channels (rgb), use average values for projected points
    if pc.shape[1] > 3:
        avg_extra = pc[:, 3:].mean(axis=0, keepdims=True)
        projected_pc_down[:, 3:] = avg_extra
    
    return projected_pc_down

def centralize_downsample(pc, pc_shape, obj_centric = True, add_bottom = False, method = 'random', debug_visualize = True):
    """
    Downsample and centralize point cloud.
    
    Args:
        pc: Point cloud of shape (N, D) where D >= 3.
            First 3 channels are xyz (centralized).
            Additional channels (e.g., rgb at channels 3:6) are preserved unchanged.
        pc_shape: Target shape (num_points, D).
        obj_centric: If True, subtract centroid from xyz.
        add_bottom: If True, add projected bottom points.
        method: Downsampling method ('random', 'fps', 'uniform').
        debug_visualize: If True, save point cloud for debugging.
        
    Returns:
        input_pc: Downsampled point cloud of shape (num_points, D)
        pc_offset: Centroid offset (3,) - only xyz offset, not full D
    """
    input_pc = np.asarray(pc)
    assert len(input_pc.shape) == 2 

    if add_bottom:
        input_pc = np.concatenate([input_pc, add_projected_point(input_pc)], axis=0)

    input_pc= downsample_pc(input_pc, pc_shape[0], method=method, debug_visualize=debug_visualize)

    if obj_centric:
        # Only centralize xyz (first 3 channels), preserve other channels (e.g., rgb)
        pc_offset = np.mean(input_pc[:, :3], axis=0)
        input_pc[:, :3] = input_pc[:, :3] - pc_offset
    else:
        pc_offset = np.zeros(3)
    return input_pc, pc_offset

def centralize_grasp( grasp, pc_offset):
    grasp[:3, 3] -= pc_offset
    return grasp

def decentralize_cond_pc(pc, pc_offset):
    pc = pc + pc_offset
    return pc

    
def decentralize_grasp(grasp, pc_offset, ref_grasp = None):
    ## if the data is grasp pose, expand the dimension 
    if len(grasp.shape) == 2:
        is_grasp_pose = True
        grasp = np.expand_dims(grasp, axis=0)
    else:
        is_grasp_pose = False

    grasp[:, :3, 3] += pc_offset
    ##below for debug, visualize right grasp rot
    if ref_grasp is not None:
        grasp[:, :3, :3] = ref_grasp
    if grasp.shape[1] ==8:
        grasp[:, 4:7, 3] += pc_offset

    ## shrink the dim 
    if is_grasp_pose:
        grasp = np.squeeze(grasp, axis=0)
    return grasp

def combined_pc_instances_and_offset(related_pc_dict, part_pc_shape, is_obj_centric=True, is_add_bottom= True, downsample_method = 'fps'):

    init_pcs = []
    pc_offsets = []
    for obj_name in related_pc_dict.keys():
        pc, pc_offset = centralize_downsample(related_pc_dict[obj_name][:, :3], part_pc_shape, obj_centric = is_obj_centric, add_bottom = is_add_bottom, method = downsample_method, debug_visualize=False)
        init_pcs.append(pc)
        pc_offsets.append(pc_offset)
    init_pc_n = np.concatenate(init_pcs, axis=0)
    init_pc_offset = (pc_offsets[0] + pc_offsets[1])/2
    return init_pc_n, init_pc_offset

def get_sg(hdf5_group, sg_name):
    sg_json = hdf5_group[sg_name][()] if sg_name in hdf5_group else None
    if sg_json is None:
        return None
    sg_str = sg_json.decode('utf-8')
    sg = nx.node_link_graph(json.loads(sg_str))
    return sg

def get_rbt_states(obs_grp, robot_names):
    data_dict = {}
    for robot_name in robot_names:
        data_dict[f'{robot_name}_joint_pos'] = obs_grp[f'{robot_name}_joint_pos'][()]
        data_dict[f'{robot_name}_eef_pos'] = obs_grp[f'{robot_name}_eef_pos'][()]
        data_dict[f'{robot_name}_eef_quat'] = obs_grp[f'{robot_name}_eef_quat'][()]
        data_dict[f'{robot_name}_gripper_qpos'] = obs_grp[f'{robot_name}_gripper_qpos'][()]

    return data_dict

def get_obj_visibility(obs_grp, obj_names):
    obj_visibility = {}
    for obj_name in obj_names:
        pc_key = f'{obj_name}_visible'
        if pc_key in obs_grp:
            obj_visibility[obj_name] = obs_grp[pc_key][()]
    return obj_visibility

def get_rbt_actions(action_arr, robot_names):
    data_dict = {}
    for robot_name in robot_names:
        rbt_idx = robot_name[-1]
        gripper_action = action_arr[:, 6+ int(rbt_idx)*7]
        data_dict[robot_name] = gripper_action

    return data_dict

def get_pc_instances(obs_grp, obj_names):
    obj_pcds = {}
    for obj_name in obj_names:
        pc_key = f'{obj_name}_point_cloud'
        if pc_key in obs_grp:
            obj_pcds[obj_name] = obs_grp[pc_key][()]

    return obj_pcds

def rotate_dataslice(data_slice):
    ## input: dataslice: dict of tensors

    yaw_rotation =  np.random.uniform(-np.pi, np.pi)
    from equibot.envs.sim_mobile.utils.transformations import euler2mat
    rot_3x3 = euler2mat([0, 0, yaw_rotation]) 
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rot_3x3
    
    data_np = to_np(data_slice)
    data_rotated = data_np.copy()
    for k, v in data_np.items():
        if k.endswith('pc'):
            pc_np = v
            rotated_pc = rotate_around_z(pc_np, yaw_rotation)
            data_rotated[k] = rotated_pc    
        elif k.endswith('eefpos'):
            grasp_np = v  ## B, 4, 4
            rotated_grasp = trans_mat[None] @ grasp_np
            data_rotated[k] = rotated_grasp
    data_tensor = to_tensor(data_rotated)
    return data_tensor

def choose_ids(traj_len, idx_list, essential_ids = None, skill_key = None):
        ## for release, only use essential ids
    if skill_key == 'release':
        selected_ids = np.random.choice(essential_ids, size=traj_len, replace=True).astype(np.int32)
        return np.sort(selected_ids).tolist()

    if essential_ids is None:
        selected_ids = np.random.choice(idx_list, size=traj_len, replace=False)
        selected_ids = list(np.sort(selected_ids.astype(np.int32)))
        return selected_ids
    
    preselected_ids = set([idx_list[0], idx_list[-1], essential_ids[0], essential_ids[-1]]) 
    remaining_ids = list(set(idx_list) - set(preselected_ids))
    other_nums = (traj_len - len(preselected_ids))
    selected_ids = np.random.choice(remaining_ids, size=other_nums, replace=False).astype(np.int32)

    ## example: if traj_len ==4, then the traj will be idx_list[0], essential_ids[0], essential_ids[-1], idx_list[-1]
    selected_ids =sorted( list(selected_ids) + list(preselected_ids))

    assert len(selected_ids) == traj_len, f"Selected ids length {len(selected_ids)} does not match traj_len {traj_len}."
    return selected_ids

## TODO: test this method
def choose_ids_rdp(traj, target_len, idx_list, essential_ids=None):
    """
    Use Ramer-Douglas-Peucker algorithm to perform trajectory simplification on traj[idx_list].
    First keeps first and last elements of idx_list and essential_ids, then applies RDP to remaining IDs.
    
    Args:
        traj: Full trajectory data
        target_len: Target number of points to select
        idx_list: List of indices to consider for selection
        essential_ids: List of essential indices that must be included
        
    Returns:
        List of selected indices that represent the simplified trajectory
    """
    import numpy as np
    from rdp import rdp
    
    if len(idx_list) <= target_len:
        return idx_list
    
    # Ensure essential_ids has at least 2 elements for first and last
    if essential_ids is None or len(essential_ids) < 2:
        essential_ids = [idx_list[0], idx_list[-1]]
    
    # Always include first and last elements from both lists
    first_idx = idx_list[0]
    last_idx = idx_list[-1]
    first_essential = essential_ids[0]
    last_essential = essential_ids[-1]
    
    # Collect IDs that must be preserved
    preserved_ids = set([first_idx, last_idx, first_essential, last_essential])
    
    # Get remaining IDs for RDP processing
    remaining_ids = [idx for idx in idx_list if idx not in preserved_ids]
    
    if len(remaining_ids) == 0:
        # If no remaining IDs, just return the preserved ones
        selected_ids = sorted(list(preserved_ids))
        # Ensure we don't exceed target_len
        if len(selected_ids) > target_len:
            selected_ids = selected_ids[:target_len]
        return selected_ids
    
    # Calculate how many additional points we can select from RDP
    remaining_slots = target_len - len(preserved_ids)
    
    if remaining_slots <= 0:
        # If we can't add more points, return preserved ones (truncated if needed)
        selected_ids = sorted(list(preserved_ids))[:target_len]
        return selected_ids
    
    # Apply RDP to the remaining trajectory segment
    traj_remaining = traj[remaining_ids]
    
    # Use adaptive epsilon to get close to remaining_slots
    epsilon_range = np.logspace(-3, 0, 10)
    
    best_indices = None
    best_count = 0
    
    for epsilon in epsilon_range:
        # Apply RDP
        simplified_points = rdp(traj_remaining, epsilon=epsilon)
        
        # Find which original indices correspond to the simplified points
        simplified_indices = []
        for point in simplified_points:
            # Find the closest original point
            distances = [np.linalg.norm(point - orig_point) for orig_point in traj_remaining]
            closest_idx = np.argmin(distances)
            simplified_indices.append(closest_idx)
        
        # Remove duplicates and sort
        simplified_indices = sorted(list(set(simplified_indices)))
        count = len(simplified_indices)
        
        # Update best if we get closer to remaining_slots
        if count <= remaining_slots and count > best_count:
            best_indices = simplified_indices
            best_count = count
        
        # If we get exactly remaining_slots, we're done. typically, we epsilon is 0.01 for 4 and 0.002 for 5
        if count == remaining_slots:
            # print(f"current epsilon: {epsilon}, count: {count}")
            break
    
    # If RDP didn't work well, sample randomly from remaining_ids
    if best_indices is None or best_count < remaining_slots:
        if len(remaining_ids) >= remaining_slots:
            best_indices = np.random.choice(remaining_ids, size=remaining_slots, replace=False).tolist()
        else:
            best_indices = remaining_ids
    
    # # Convert back to original indices and combine with preserved ones
    # # best_indices contains indices relative to traj_remaining, so we need to map them to actual remaining_ids
    # if best_indices and isinstance(best_indices[0], int) and best_indices[0] < len(remaining_ids):
    #     # best_indices contains positions in traj_remaining, map to actual remaining_ids
    #     selected_remaining = [remaining_ids[i] for i in best_indices]
    # else:
    #     # best_indices already contains actual IDs (from random sampling fallback)
    #     selected_remaining = best_indices if best_indices else []
    
    selected_ids = sorted(list(preserved_ids) + list(best_indices))
    
    # Ensure we have exactly target_len points
    if len(selected_ids) > target_len:
        # Keep first and last, then sample the rest
        final_ids = [selected_ids[0], selected_ids[-1]]
        remaining_slots = target_len - 2
        if remaining_slots > 0:
            middle_ids = [idx for idx in selected_ids[1:-1] if idx not in [selected_ids[0], selected_ids[-1]]]
            if len(middle_ids) >= remaining_slots:
                additional = np.random.choice(middle_ids, size=remaining_slots, replace=False)
                final_ids.extend(additional.tolist())
            else:
                final_ids.extend(middle_ids)
        selected_ids = final_ids
    
    assert len(selected_ids) == target_len, f"Selected indices length {len(selected_ids)} does not match target_len {target_len}"
    
    return selected_ids


def render_trajectory(pc, eef_poses,  gripper_values=None, title = 'prediction', max_resolution=400, dpi=100):
    """
    Render point cloud and full trajectory of end-effector poses using matplotlib.
    
    Args:
        pc: Point cloud data (N, 3) or (N, 6)
        eef_poses: End-effector poses for all timesteps (T, 4, 4)
        gripper_values: Gripper values for each timestep (T,) - if provided, colors trajectory based on gripper state
        title: Title for the plot
        max_resolution: Maximum resolution (width or height) of the output image
        dpi: Dots per inch for the figure
        
    Returns:
        rendered_image: RGB image as numpy array
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for headless rendering
    
    # Calculate figure size to limit resolution
    # max_resolution = max(width, height) in pixels
    # figure_size = max_resolution / dpi
    max_fig_size = max_resolution / dpi
    fig_width = max_fig_size * 1.25  # 10/8 aspect ratio
    fig_height = max_fig_size * 1.0
    
    # Create figure with calculated size
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud with optional color
    if pc.shape[-1] == 6:
        pc_color = pc[:, 3:]
        # Normalize color to [0, 1] if in [0, 255] range
        if pc_color.max() > 1.0:
            pc_color = pc_color / 255.0
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc_color, s=max_fig_size, alpha=0.6, label='Point Cloud')
    else:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='red', s=max_fig_size, alpha=0.6, label='Point Cloud')
    
    # Plot trajectory
    trajectory_points = []
    
    # Determine colors based on gripper values
    if gripper_values is not None:
        # Color based on gripper state: red for closed (negative), blue for open (positive)
        colors = []
        for gripper_val in gripper_values:
            if gripper_val > 0:  # Closed gripper
                colors.append('red')
            else:  # Open gripper
                colors.append('blue')
    else:
        # Default color gradient if no gripper values provided
        colors = plt.cm.viridis(np.linspace(0, 1, len(eef_poses)))
    
    for t, pose in enumerate(eef_poses):
        # Extract position
        pos = pose[:3, 3]
        trajectory_points.append(pos)
        
        # Plot coordinate frame
        origin = pos
        x_axis = pose[:3, 0] * 0.05  # Scale down the axes
        y_axis = pose[:3, 1] * 0.05
        z_axis = pose[:3, 2] * 0.05
        
        # Draw coordinate axes
        ax.quiver(origin[0], origin[1], origin[2], 
                 x_axis[0], x_axis[1], x_axis[2], 
                 color='red', alpha=0.8, length=0.5+t*0.05)
        ax.quiver(origin[0], origin[1], origin[2], 
                 y_axis[0], y_axis[1], y_axis[2], 
                 color='green', alpha=0.8, length=0.5+t*0.05)
        ax.quiver(origin[0], origin[1], origin[2], 
                 z_axis[0], z_axis[1], z_axis[2], 
                 color='blue', alpha=0.8, length=0.5+t*0.05)
        
        # Add colored sphere for each pose based on gripper state
        if gripper_values is not None:
            ax.scatter(pos[0], pos[1], pos[2], c=colors[t], s=max_fig_size, alpha=0.8)
        else:
            ax.scatter(pos[0], pos[1], pos[2], c=[colors[t]], s=max_fig_size, alpha=0.8)
    
    # Connect trajectory points with line
    if len(trajectory_points) > 1:
        trajectory_points = np.array(trajectory_points)
        ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
               'y-', linewidth=2, alpha=0.7, label='Trajectory')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # title = f'{skill_name} - {vis_type}'
    # if gripper_values is not None:
    #     title += ' (Red=Closed, Blue=Open)'
    ax.set_title(title)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    ## add eefpos for a broader range
    for t, pose in enumerate(eef_poses):
        pos = pose[:3, 3]
        if pc.shape[-1] == 6:
            pos = np.concatenate([pos, [0,0, 0]], axis=0)
        pc = np.concatenate([pc, pos[None]], axis=0)

    # Get the data ranges
    pc_x_range = pc[:, 0].max() - pc[:, 0].min()
    pc_y_range = pc[:, 1].max() - pc[:, 1].min()
    pc_z_range = pc[:, 2].max() - pc[:, 2].min()
    
    # Get trajectory ranges
    if len(trajectory_points) > 0:
        traj_points = np.array(trajectory_points)
        traj_x_range = traj_points[:, 0].max() - traj_points[:, 0].min()
        traj_y_range = traj_points[:, 1].max() - traj_points[:, 1].min()
        traj_z_range = traj_points[:, 2].max() - traj_points[:, 2].min()
    else:
        traj_x_range = traj_y_range = traj_z_range = 0
    
    # Find the maximum range across all dimensions
    max_range = max(pc_x_range, pc_y_range, pc_z_range, traj_x_range, traj_y_range, traj_z_range)
    
    # Add some padding (10% of max range)
    padding = max_range * 0.1
    
    # Set the same limits for all axes
    if max_range > 0:
        # Get the center of all data
        all_x = np.concatenate([pc[:, 0], np.array(trajectory_points)[:, 0] if len(trajectory_points) > 0 else []])
        all_y = np.concatenate([pc[:, 1], np.array(trajectory_points)[:, 1] if len(trajectory_points) > 0 else []])
        all_z = np.concatenate([pc[:, 2], np.array(trajectory_points)[:, 2] if len(trajectory_points) > 0 else []])
        
        x_center = np.mean(all_x)
        y_center = np.mean(all_y)
        z_center = np.mean(all_z)
        
        # Set limits with equal range
        half_range = max_range / 2 + padding
        ax.set_xlim(x_center - half_range, x_center + half_range)
        ax.set_ylim(y_center - half_range, y_center + half_range)
        ax.set_zlim(z_center - half_range, z_center + half_range)
    
    # Add legend
    ax.legend()
    
    # Auto-adjust view to fit data
    ax.autoscale_view()
    
    # Tight layout
    plt.tight_layout()
    
    # Convert to image array
    fig.canvas.draw()
    rendered_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    rendered_image = rendered_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    rendered_image = rendered_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Close figure to free memory
    plt.close(fig)
    
    return rendered_image


def vis_metric_imgs(metrics, save_name = "eval_debug.png"):
    img_keys = [k for k in metrics.keys() if k.endswith("image")]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4 * len(img_keys), 4))
    for i, img_key in enumerate(img_keys, start=1):
        img = metrics[img_key]

        # If the image is a tensor, convert to numpy
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):  # C,H,W -> H,W,C
            img = img.transpose(1, 2, 0)

        plt.subplot(1, len(img_keys), i)
        plt.imshow(img)
        plt.title(img_key)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_name)