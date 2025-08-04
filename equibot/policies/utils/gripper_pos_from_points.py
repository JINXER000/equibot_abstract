#import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from misc import matrix_to_rotation_6d, rotation_6d_to_matrix

original_gripper_pcd = np.array([[ 0.10432111,  0.00228697,  0.8474241 ],
       [ 0.12816067, -0.04368229,  0.8114649 ],
       [ 0.08953098,  0.0484529 ,  0.80711854],
       [ 0.11198021,  0.00245327,  0.7828771 ]])

original_gripper_pos = np.array([0.1119802 , 0.00245327, 0.78287711])
original_gripper_orn = np.array([0.97841681, 0.19802945, 0.0581003 , 0.01045192])


# original_gripper_pcd = np.array([[ 1.00000000e+00, -9.92297486e-19, -3.70473796e-18],
#  [ 2.66000054e-02,  4.99999920e-02, -1.26660558e-18],
#  [ 2.66000374e-02, -5.00001931e-02, -1.30920958e-08],
#  [ 0.00000000e+00, 0.00000000e+00,  0.00000000e+00]])

# def matrix_to_rotation_6d_numpy(R_batch):
#     """
#     Converts a batch of 3x3 rotation matrices to 6D continuous rotation representations.
    
#     Args:
#         R_batch: ndarray of shape (B, 3, 3) — batch of rotation matrices.

#     Returns:
#         ndarray of shape (B, 6) — batch of 6D rotation representations.
#     """
#     # Sanity check
#     assert R_batch.ndim == 3 and R_batch.shape[1:] == (3, 3), "Input must be (B, 3, 3) shape"
    
#     # Extract first two columns
#     col1 = R_batch[:, :, 0]  # (B, 3)
#     col2 = R_batch[:, :, 1]  # (B, 3)
    
#     # Concatenate into 6D
#     rot_6d = np.concatenate([col1, col2], axis=-1)  # (B, 6)
    
#     return rot_6d

# def rotation_transfer_6D_to_matrix(x):
#     """
#     Converts a 6D rotation representation to a 3x3 rotation matrix using Gram-Schmidt.
    
#     Args:
#         x (np.ndarray): shape (6,) — 6D rotation representation (first two columns of rotation matrix)
    
#     Returns:
#         np.ndarray: shape (3, 3) — 3x3 rotation matrix
#     """
#     assert x.shape == (6,), f"Expected shape (6,), got {x.shape}"

#     a1 = x[0:3]
#     a2 = x[3:6]

#     b1 = a1 / np.linalg.norm(a1)

#     dot = np.dot(b1, a2)
#     a2_proj = dot * b1
#     a2_orth = a2 - a2_proj
#     b2 = a2_orth / np.linalg.norm(a2_orth)

#     b3 = np.cross(b1, b2)

#     R = np.stack([b1, b2, b3], axis=1)  # shape (3, 3), columns are basis vectors

#     return R


def compute_plane_normal(gripper_pcd):
    x1 = gripper_pcd[0]
    x2 = gripper_pcd[1]
    x4 = gripper_pcd[3]
    v1 = x2 - x1
    v2 = x4 - x1
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)



def quaternion_to_rotation_matrix(quat):
    rotation = R.from_quat(quat)
    return rotation.as_matrix()

def rotation_matrix_to_quaternion(R_opt):
    rotation = R.from_matrix(R_opt)
    return rotation.as_quat()

def rotation_matrix_from_vectors(v1, v2):
    """
    Find the rotation matrix that aligns v1 to v2
    :param v1: A 3d "source" vector
    :param v2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to v1, aligns it with v2.
    """
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


def _get_gripper_pos_orient_from_4_points(original_gripper_pcd, gripper_pcd, original_gripper_pos, original_gripper_orn, original_gripper_normal ):
    normal = compute_plane_normal(gripper_pcd)
    R1 = rotation_matrix_from_vectors(original_gripper_normal, normal)  ## rotation from plane to plane
    v1 = original_gripper_pcd[3] - original_gripper_pcd[0]
    v2 = gripper_pcd[3] - gripper_pcd[0]
    v1_prime = np.dot(R1, v1)
    R2 = rotation_matrix_from_vectors(v1_prime, v2)  ## rotation in-plane
    R = np.dot(R2, R1)
    gripper_pos = original_gripper_pos + gripper_pcd[3] - original_gripper_pcd[3]
    original_R = quaternion_to_rotation_matrix(original_gripper_orn)
    R = np.expand_dims(np.dot(R, original_R), axis = 0)
    #print("RRRRRRRRRR", R)
    # gripper_orn = matrix_to_rotation_6d_numpy(R.swapaxes(1, 2))
    # gripper_orn = matrix_to_rotation_6d_numpy(R)
    gripper_orn = matrix_to_rotation_6d(torch.tensor(R)).detach().numpy()

    return gripper_pos, np.squeeze(gripper_orn, axis = 0)



def compute_plane_normal(gripper_pcd):
    x1 = gripper_pcd[0]
    x2 = gripper_pcd[1]
    x4 = gripper_pcd[3]
    v1 = x2 - x1
    v2 = x4 - x1
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

def get_gripper_pos_orient_from_4_points_torch(gripper_pcd):
    #import pdb; pdb.set_trace();
    original_gripper_normal = compute_plane_normal(original_gripper_pcd)
    #print("HEREEEEEEEE", gripper_pcd.shape, original_gripper_pcd.shape)
    #gripper_pcd = gripper_pcd.T
    gripper_pos, gripper_orn = _get_gripper_pos_orient_from_4_points(original_gripper_pcd, gripper_pcd, original_gripper_pos, original_gripper_orn, original_gripper_normal)
    return np.concatenate((gripper_pos.reshape(3), gripper_orn.reshape(6)))


def get_points_from_pos_rotation_matrix(pos, orient):

    # absolute_rotation = rotation_transfer_6D_to_matrix(orient.numpy())
    absolute_rotation = rotation_6d_to_matrix(orient.reshape(1, 6)).squeeze(0).numpy()

    original_R = quaternion_to_rotation_matrix(original_gripper_orn)
    rotation_transfer = absolute_rotation @ original_R.T
    original_pcd = original_gripper_pcd - original_gripper_pcd[3]
    rotated_pcd = np.dot(original_pcd, rotation_transfer.T)
    gripper_pcd = rotated_pcd + pos
    return gripper_pcd

def is_coplanar(pcd):

    A, B, C, D = pcd

    v1 = B - A
    v2 = C - A
    v3 = D - A

    volume = np.abs(np.dot(v1, np.cross(v2, v3))) / 6.0

    return np.isclose(volume, 0.0, atol=1e-4)



def plot_gripper_pcd(pcd, title="Gripper Point Cloud", show_axes=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter points
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='r', s=50)

    # Connect the points (optional)
    for i, p in enumerate(pcd):
        ax.text(p[0], p[1], p[2], f'{i}', fontsize=10)

    if show_axes:
        # Draw coordinate axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.05)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.05)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.05)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # Equal aspect
    plt.show()
    plt.savefig(f"{title}.png", bbox_inches='tight', dpi=300)



def test_np_conversion():

    gripper_pcd = np.array([[ 0.7666899, -0.16390935, 0.44825187],
        [ 0.8285312, -0.16042456, 0.4600023 ],
        [ 0.7625436, -0.1860767, 0.38937932],
        [ 0.8155203, -0.17972143, 0.40836987]])

    cur_pcd_coplanar = is_coplanar(gripper_pcd)

    gripper_10d = get_gripper_pos_orient_from_4_points_torch(gripper_pcd)
    print(gripper_10d)
    gripper_pcd_reconstructed = get_points_from_pos_rotation_matrix(gripper_10d[:3], torch.tensor(gripper_10d[3:]))
    print(gripper_pcd_reconstructed)
    error = np.linalg.norm(gripper_pcd - gripper_pcd_reconstructed, axis=1)
    print("Reconstruction error per point:", error)
    print("Max error:", np.max(error))

def test_torch_conversion(raw_pcd, pred_pcd):

    cur_pcd_coplanar = is_coplanar(pred_pcd)
    print("cur_pcd_coplanar", cur_pcd_coplanar)

    gripper_pcd_tensor = torch.tensor(pred_pcd).reshape(1, 1, 4, 3)
    from misc import convert_trans_to_4pts, convert_4pts_to_trans
    gripper_trans = convert_4pts_to_trans(gripper_pcd_tensor, raw_pcd)
    print("pred_trans", gripper_trans)
    gripper_pcd_reconstructed = convert_trans_to_4pts(gripper_trans, raw_pcd)
    print("pred_pcd_reconstructed", gripper_pcd_reconstructed)

if __name__ == "__main__":
    # test_np_conversion()

    # original_R = quaternion_to_rotation_matrix(original_gripper_orn)
    # raw_pcd = (original_gripper_pcd - original_gripper_pos) @ original_R.T
    # print("raw_pcd", raw_pcd)

    # pred_pcd = np.array([[ 0.7666899, -0.16390935, 0.44825187],
    #     [ 0.8285312, -0.16042456, 0.4600023 ],
    #     [ 0.7625436, -0.1860767, 0.38937932],
    #     [ 0.8155203, -0.17972143, 0.40836987]])
    
    raw_pcd = np.array([[0.07, 0.01, 0], 
                        [0.02, -0.04, 0],
                        [0.02, 0.05, 0],
                        [0,0,0]])
    from scipy.spatial.transform import Rotation as R
    rot60 = R.from_euler('y', 60, degrees=True).as_matrix()
    pred_pcd = raw_pcd @ rot60.T
    print("pred_pcd", pred_pcd)
    test_torch_conversion(raw_pcd, pred_pcd)