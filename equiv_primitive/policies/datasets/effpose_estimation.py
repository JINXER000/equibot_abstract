

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points as fps
from pytorch3d.ops.points_alignment import iterative_closest_point, SimilarityTransform 

import pathlib
import os
import open3d as o3d
from equiv_primitive.policies.utils.misc import rotate_around_z
from equiv_primitive.policies.vision.vdgcnn_encoder import VecDGCNN_att_frozen

def transformation_residuals(x1, x2, R, t):
    """
    Computer the pointwise residuals based on the estimated transformation paramaters
    
    Args:
        x1  (torch array): points of the first point cloud [b,n,3]
        x2  (torch array): points of the second point cloud [b,n,3]
        R   (torch array): estimated rotation matrice [b,3,3]
        t   (torch array): estimated translation vectors [b,3,1]
    Returns:
        res (torch array): pointwise residuals (Eucledean distance) [b,n,1]
    """
    x2_reconstruct = torch.matmul(R, x1.transpose(1, 2)) + t 

    res = torch.norm(x2_reconstruct.transpose(1, 2) - x2, dim=2)

    return res

def kabsch_transformation_estimation(x1, x2, weights=None, normalize_w = True, eps = 1e-7, best_k = 0, w_threshold = 0):
    """
    Torch differentiable implementation of the weighted Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm). Based on the correspondences and weights calculates
    the optimal rotation matrix in the sense of the Frobenius norm (RMSD), based on the estimate rotation matrix is then estimates the translation vector hence solving
    the Procrustes problem. This implementation supports batch inputs.

    Args:
        x1            (torch array): points of the first point cloud [b,n,3]
        x2            (torch array): correspondences for the PC1 established in the feature space [b,n,3]
        weights       (torch array): weights denoting if the coorespondence is an inlier (~1) or an outlier (~0) [b,n]
        normalize_w   (bool)       : flag for normalizing the weights to sum to 1
        best_k        (int)        : number of correspondences with highest weights to be used (if 0 all are used)
        w_threshold   (float)      : only use weights higher than this w_threshold (if 0 all are used)
    Returns:
        rot_matrices  (torch array): estimated rotation matrices [b,3,3]
        trans_vectors (torch array): estimated translation vectors [b,3,1]
        res           (torch array): pointwise residuals (Eucledean distance) [b,n]
        valid_gradient (bool): Flag denoting if the SVD computation converged (gradient is valid)

    """
    if weights is None:
        weights = torch.ones(x1.shape[0],x1.shape[1]).type_as(x1).to(x1.device)

    if normalize_w:
        sum_weights = torch.sum(weights,dim=1,keepdim=True) + eps
        weights = (weights/sum_weights)

    weights = weights.unsqueeze(2)

    if best_k > 0:
        indices = np.argpartition(weights.cpu().numpy(), -best_k, axis=1)[0,-best_k:,0]
        weights = weights[:,indices,:]
        x1 = x1[:,indices,:]
        x2 = x2[:,indices,:]

    if w_threshold > 0:
        weights[weights < w_threshold] = 0


    x1_mean = torch.matmul(weights.transpose(1,2), x1) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)
    x2_mean = torch.matmul(weights.transpose(1,2), x2) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)

    x1_centered = x1 - x1_mean
    x2_centered = x2 - x2_mean

    weight_matrix = torch.diag_embed(weights.squeeze(2))

    cov_mat = torch.matmul(x1_centered.transpose(1, 2),
                           torch.matmul(weight_matrix, x2_centered))

    try:
        u, s, v = torch.svd(cov_mat)
    except Exception as e:
        r = torch.eye(3,device=x1.device)
        r = r.repeat(x1_mean.shape[0],1,1)
        t = torch.zeros((x1_mean.shape[0],3,1), device=x1.device)

        res = transformation_residuals(x1, x2, r, t)

        return r, t, res, True

    tm_determinant = torch.det(torch.matmul(v.transpose(1, 2), u.transpose(1, 2)))

    determinant_matrix = torch.diag_embed(torch.cat((torch.ones((tm_determinant.shape[0],2),device=x1.device), tm_determinant.unsqueeze(1)), 1))

    rotation_matrix = torch.matmul(v,torch.matmul(determinant_matrix,u.transpose(1,2)))

    # translation vector
    translation_matrix = x2_mean.transpose(1,2) - torch.matmul(rotation_matrix,x1_mean.transpose(1,2))

    # Residuals
    res = transformation_residuals(x1, x2, rotation_matrix, translation_matrix)

    return rotation_matrix, translation_matrix, res, False


def encode(encoder, x, use_double=False):
    input_pcl = x.double() if use_double else x
    B, _, N = input_pcl.shape
    device = input_pcl.device

    # normalize the point clouds: centriod and scale
    centroid = input_pcl.mean(-1) # B,3
    input_pcl = input_pcl - centroid[..., None]

    # scale initialization
    dist = torch.cdist(input_pcl.transpose(-1,-2), input_pcl.transpose(-1,-2))
    scale_0 = dist.view(B, -1).topk(5, dim=-1)[0].mean(-1)
    input_pcl = input_pcl / scale_0[:,None,None]

    # encoding
    encoder_ret = encoder(input_pcl)

    if len(encoder_ret) == 4:
        center_pred, pred_scale, pred_so3_feat, pred_inv_feat = encoder_ret
        centroid = center_pred.squeeze(1) + centroid
        scale = scale_0 * pred_scale
    else:
        pred_scale, pred_so3_feat, pred_inv_feat = encoder_ret
        scale = scale_0 * pred_scale
    
    embedding = {
        "z_so3": pred_so3_feat,
        "z_inv": pred_inv_feat,
        "s": scale,
        "t": centroid.unsqueeze(1),
    }

    return embedding
    
def solve_pairwise_registration(encoder, pc1_full, pc2_full, n_repeat = 1, n_input_point = 512):
    '''
    Solve the 3d rigid transformation between pc1 and pc2, transform direction: pc1 -> pc2
    Args: 
        pc1: tensor (1, N, 3)
        pc2: tensor (1, M, 3)
        optim: use optimization (bool)
    Return:
        R: tensor (1, 3, 3)
        t: tensor (1, 3, 1)
    '''

    pc1, _ = fps(pc1_full.repeat_interleave(n_repeat, dim=0), K=n_input_point)
    pc2, _ = fps(pc2_full.repeat_interleave(n_repeat, dim=0), K=n_input_point)
    
    with torch.no_grad():
        code1 = encode(encoder, pc1.transpose(-1,-2))
        code2 = encode(encoder, pc2.transpose(-1,-2))

    code1_se3 = code1['z_so3'] + code1['t']
    code2_se3 = code2['z_so3'] + code2['t']
    R, t, _, _ = kabsch_transformation_estimation(code1_se3, code2_se3)
    
        # use icp refinement
    s0 = torch.tensor([1]).float().cuda()
    icp_solution  = iterative_closest_point(pc1, pc2, init_transform=SimilarityTransform(R.transpose(-1,-2), t.squeeze(2), s0))
    R, t, _ = icp_solution[3]

    R = R.transpose(-1, -2)
    t = t.unsqueeze(2)

    return R, t

def debug_and_save(start_pc, end_pc, R, t):

    ## save as ply
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(start_pc)

    dataset_path = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
    o3d.io.write_point_cloud(os.path.join(dataset_path, 'start_pc.ply'), pcd)

    pcd.points = o3d.utility.Vector3dVector(end_pc)
    o3d.io.write_point_cloud(os.path.join(dataset_path, 'end_pc.ply'), pcd)

    from scipy.spatial.transform import Rotation
    rot_mat = Rotation.from_matrix(R.squeeze().cpu().numpy())
    euler_angles = rot_mat.as_euler('zyx', degrees=True)

    # Format the Euler angles to two decimal places
    formatted_angles = [f"{angle:.2f}" for angle in euler_angles]

    print("Yaw (ψ):", formatted_angles[0], "Pitch (θ):", formatted_angles[1], "Roll (ϕ):", formatted_angles[2])

    print('The estimated rotation matrix is: ', R)
    print('The estimated translation vector is: ', t)

## TODO: integrate the pose estimation into the data postprocessing
if __name__ == '__main__':

    dataset_path = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
    ply_path = os.path.join(dataset_path, 'tape_OOD.ply')

    pcd = o3d.io.read_point_cloud(ply_path)
    input_pc = np.asarray(pcd.points)

    yaw_rot = np.pi/3
    rotated_pc = rotate_around_z(input_pc, yaw_rot)
    
    # load the pretrained encoder
    dataset_path = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
    w_enc_path = os.path.join(dataset_path, 'pretrained', 'mugs.pt')
    encoder = VecDGCNN_att_frozen(preload_path= w_enc_path).cuda()

    pc1 = torch.tensor(input_pc).unsqueeze(0).float().cuda()
    pc2 = torch.tensor(rotated_pc).unsqueeze(0).float().cuda()

    R, t = solve_pairwise_registration(encoder, pc1, pc2)

    debug_and_save(input_pc, rotated_pc, R, t)


