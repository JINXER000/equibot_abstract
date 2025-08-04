import numpy as np
import torch
import torch.nn.functional as F
import pathlib
EQUIBOT_PATH = pathlib.Path(__file__).parent.parent.parent.parent.absolute()

def to_torch(batch, device):    return {k: v.to(device) for k, v in batch.items()}

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
    # Check if the input points have the correct shape (N, 3)
    assert (len(points.shape) == 1 and len(points) == 3) or points.shape[-1] == 3
    p_shape = points.shape
    points = points.reshape(-1, 3) - center[None]

    # Create the rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
    )

    # Apply the rotation to all points using matrix multiplication
    rotated_points = np.dot(points, rotation_matrix.T) * scale[None] + center[None]
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
    else:
        raise ValueError(f"Agent with name [{agent_name}] not found.")



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
    return ''.join([chr(int(code)) for code in tensor.tolist()])

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
    if pc.shape[0] < num_points* 0.3:
        raise ValueError('Input pc shape is not enough points!')
    elif pc.shape[0] < num_points:
        random_repeated_indices = np.random.choice(pc.shape[0], num_points - pc.shape[0], replace=True)
        pc = np.concatenate([pc, pc[random_repeated_indices]], axis=0)
        return pc
    elif pc.shape[0] == num_points:
        return pc

    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
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

    ## save the pc
    if debug_visualize:
        o3d.io.write_point_cloud(f'{method}_pc.ply', pcd_down)

    return np.asarray(pcd_down.points)

def add_projected_point(pc, num_ratio = 0.5):
    """
    Project the point cloud onto the plane at the minimum z value, then downsample to 100 points.
    Args:
        pc (np.ndarray): Input point cloud of shape (N, 3)
    Returns:
        np.ndarray: Downsampled projected point cloud of shape (100, 3)
    """
    # Find the minimum z value
    min_z = np.min(pc[:, 2])
    # Project all points onto the plane z = min_z
    projected_pc = pc.copy()
    projected_pc[:, 2] = min_z
    
    num_points = int(pc.shape[0] * num_ratio)
    projected_pc_down = downsample_pc(projected_pc, num_points, method = 'random')
    return projected_pc_down

def centralize_downsample(pc, pc_shape, obj_centric = True, add_bottom = False, method = 'random', debug_visualize = True):
    input_pc = np.asarray(pc)
    assert len(input_pc.shape) == 2 

    if add_bottom:
        input_pc = np.concatenate([input_pc, add_projected_point(input_pc)], axis=0)

    input_pc= downsample_pc(input_pc, pc_shape[0], method=method, debug_visualize=debug_visualize)

    if obj_centric:
        pc_offset = np.min(input_pc, axis=0)
        input_pc = input_pc - pc_offset
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

def render_trajectory(pc, eef_poses, skill_name, gripper_values=None, show_window=True):
    """
    Render point cloud and full trajectory of end-effector poses using matplotlib.
    
    Args:
        pc: Point cloud data (N, 3)
        eef_poses: End-effector poses for all timesteps (T, 4, 4)
        skill_name: Name of the skill for visualization
        gripper_values: Gripper values for each timestep (T,) - if provided, colors trajectory based on gripper state
        show_window: Whether to show the visualization window (ignored for headless rendering)
        
    Returns:
        rendered_image: RGB image as numpy array
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for headless rendering
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='red', s=30, alpha=0.6, label='Point Cloud')
    
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
                 color='red', alpha=0.8, length=0.5)
        ax.quiver(origin[0], origin[1], origin[2], 
                 y_axis[0], y_axis[1], y_axis[2], 
                 color='green', alpha=0.8, length=0.5)
        ax.quiver(origin[0], origin[1], origin[2], 
                 z_axis[0], z_axis[1], z_axis[2], 
                 color='blue', alpha=0.8, length=0.5)
        
        # Add colored sphere for each pose based on gripper state
        if gripper_values is not None:
            ax.scatter(pos[0], pos[1], pos[2], c=colors[t], s=50, alpha=0.8)
        else:
            ax.scatter(pos[0], pos[1], pos[2], c=[colors[t]], s=50, alpha=0.8)
    
    # Connect trajectory points with line
    if len(trajectory_points) > 1:
        trajectory_points = np.array(trajectory_points)
        ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
               'g-', linewidth=2, alpha=0.7, label='Trajectory')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title = f'{skill_name} - Point Cloud and Trajectory'
    if gripper_values is not None:
        title += ' (Red=Closed, Blue=Open)'
    ax.set_title(title)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set equal scales for all dimensions
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
