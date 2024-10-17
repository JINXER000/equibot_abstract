import numpy as np
import torch
import torch.nn.functional as F

def to_torch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


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
    agent_name = cfg.agent.agent_name
    if agent_name == "aloha":
        from equibot.policies.datasets.abstract_dataset import ALOHAPoseDataset
        return ALOHAPoseDataset(cfg.data.dataset, mode)
    elif agent_name == "compaloha":
        from equibot.policies.datasets.dual_abs_dataset import DualAbsDataset
        return DualAbsDataset(cfg.data.dataset, mode)
    else:
        from equibot.policies.datasets.dataset import BaseDataset
        return BaseDataset(cfg.data.dataset, mode)


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

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

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
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


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

class ActionSlice(object):
    def __init__(self, mode = "separated"):
        self.mode = mode
        if mode == "separated":
            self.data = {'left_jpose': None, 'right_jpose': None, \
                         'left_grasp': None, 'right_grasp': None}
            self.ee_dof = 7
        else:
            self.data = {'jpose': None, 'grasp': None}
            self.ee_dof = 12

    def update(self, key, value):
        self.data[key] = value

    def get(self, key):
        if not key in self.data:
            return None
        return self.data[key]
    

