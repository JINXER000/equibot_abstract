import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import namedtuple
from equibot.policies.utils.constants import qpos_to_eepose

from equibot.policies.vision.vdgcnn_encoder import VecDGCNN_att_frozen
from equibot.policies.datasets.effpose_estimation import solve_pairwise_registration, debug_and_save
from equibot.policies.utils.misc import to_torch, rotate_observation, rotate_around_z, to_tensor, to_np, convert_trans_to_vec, convert_vec_to_trans, EQUIBOT_PATH


import hydra
import sys
sys.path.append('/home/user/yzchen_ws/TAMP-ubuntu22/pddlstream_aloha')
# sys.path.append('/mnt/TAMP/interbotix_ws/src/pddlstream_aloha')
# sys.path.append('/home/xuhang/interbotix_ws/src/pddlstream_aloha')
# from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose
# from examples.pybullet.aloha_real.scripts.aloha_tamp_constants import qpos_to_eepose

import pathlib
EQUIBOT_PATH = pathlib.Path(__file__).parent.parent.parent.parent.absolute()


feature_tuple = namedtuple('feature_tuple', ['dim', 'start', 'end'])

def downsample_pc(pc, num_points):
    if pc.shape[0] > num_points:
        sampled_indices = np.random.choice(pc.shape[0], num_points, replace=False)
        pc = pc[sampled_indices]
    elif pc.shape[0] < num_points:
        if pc.shape[0] < num_points *0.5:
            raise ValueError('Input pc shape is not enough points!')
        else:
            random_repeated_indices = np.random.choice(pc.shape[0], num_points - pc.shape[0], replace=True)
            pc = np.concatenate([pc, pc[random_repeated_indices]], axis=0)
    return pc

def save_dbg_pc(pc):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud('debug_pc.ply', pcd)

class ALOHAPoseDataset(Dataset):
    def __init__(self, cfg, mode, transform=None, pre_transform=None, pre_filter=None, force_process = False, **kwargs):
        super().__init__()
        self.mode = mode
        self.dir_name = cfg.path
        self.root = self.dir_name
        # self.symb_mask = cfg.symb_mask
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.composed_inference = False

        # self.mj_offset = np.array([0.0, 0.5, 0.0])
        self.pc_shape = (cfg.num_points, 3)
        self.has_eff_list = cfg.has_eff_list
        self.has_eff = True in self.has_eff_list
        # self.is_mj = ('mj' in cfg.dataset_type)

        self.is_obj_centric = cfg.is_obj_centric

        self.num_eef = cfg.num_eef
        self.dof = cfg.dof

        # self.process_select(cfg,**kwargs)
        if mode == 'train' or force_process == True:
            # Process the data
            print('Processing dataset...')
            self.process_select(cfg,**kwargs)
        else:
            print('Loading dataset...')

        
        if mode != 'inference':
            # Load processed data
            self.data, self.slices = torch.load(self.processed_file_path)

    @property
    def raw_file_names(self):
        return os.listdir(os.path.join(self.root, 'raw'))

    @property
    def processed_file_path(self):
        return os.path.join(self.root, 'processed', 'data.pt')

    def centralize_cond_pc(self,  pc, obj_centric = True):
        input_pc = np.asarray(pc)
        assert len(input_pc.shape) == 2 
        input_pc= downsample_pc(input_pc, self.pc_shape[0])
        
        ## get pc in the world frame (the origin in the middle of robots)
        # if self.is_mj:
        #     input_pc = input_pc - self.mj_offset
        
        if obj_centric:
            pc_offset = np.min(input_pc, axis=0)
            input_pc = input_pc - pc_offset
        else:
            pc_offset = np.zeros(3)
        return input_pc, pc_offset
    
    def centralize_grasp(self, grasp, pc_offset):
        grasp[:3, 3] -= pc_offset
        if self.has_eff:
            grasp[4:7, 3] -= pc_offset
        return grasp
    
    def decentralize_cond_pc(self,  pc, pc_offset):
        pc = pc + pc_offset
        return pc
    
      
    def decentralize_grasp(self,  grasp, pc_offset, ref_grasp = None, **kwargs):
        grasp[:3, 3] += pc_offset
        ##below for debug, visualize right grasp rot
        if ref_grasp is not None:
            grasp[:3, :3] = ref_grasp
        if grasp.shape[0] ==8:
            grasp[4:7, 3] += pc_offset
        return grasp
    
    def process_select(self, cfg, **kwargs):

        # self.norm_stat_dict = nn.ParameterDict({
        #     'jpose': None,
        #     'pc': None,
        #     'grasp': None,
        # })
        if cfg.dataset_type == 'sam_predeff':
            self.process_sam_predeff(cfg)
        elif cfg.dataset_type == 'npz':
            self.process_riemanngrasp(cfg)
        elif cfg.dataset_type == 'txt':
            self.process_txt(cfg)
        elif cfg.dataset_type == 'hdf5_predeff':
            self.process_50demos_predeff(cfg)
        elif cfg.dataset_type == 'hdf5_mini':
            self.process_hdf5_mini(cfg,**kwargs)
        else:
            raise NotImplementedError(f'Dataset type {cfg.dataset_type} not implemented!')

        

    def process_txt(self, cfg):
        print('Processing dataset...')
        data_list = []
        raw_files = self.raw_file_names

        conditional_pc = None
        grasp_trans = None
        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]
            # joint pose
            if  file_name == 'transfer_jpose.txt':
                txt_path = os.path.join(self.root, 'raw', file_name)
                jpos_mat = np.loadtxt(txt_path)  # contain 50 demos

                # produce_demos()
                for i in tqdm(range(jpos_mat.shape[0])): 
                    if self.pre_transform is not None:
                        raise NotImplementedError('Should implement jpose to eepose')
                        data = self.pre_transform(jpos_mat[i])

                    line_rcd = torch.tensor(jpos_mat[i], dtype=torch.float32) # t, arm_left_6d, arm_right_6d
                    joint_pose = line_rcd[1:].reshape(-1, cfg.dof).unsqueeze(0)
                    demo_t = line_rcd[0].unsqueeze(0)
                    # mask_data = self.mask[:joint_pose.shape[1]].unsqueeze(0)
                    # # NOTE: unsqueeze(0) is important, making each x into a shape of [1, 12]
                    # data = {'demo_t': demo_t, 'jpose': joint_pose, 'mask': mask_data}
                    data = {'demo_t': demo_t, 'jpose': joint_pose}
                    data_list.append(data)
            elif  file_name  == 'graspobj_4.ply':
                ply_path = os.path.join(self.root, 'raw', file_name)
                import open3d as o3d
                conditional_pc = o3d.io.read_point_cloud(ply_path)
                conditional_pc = np.asarray(conditional_pc.points)

                conditional_pc = downsample_pc(conditional_pc, cfg.num_points)

            elif  file_name == 'graspPose_4.npz': # as a dummy input of vnn
                npz_path = os.path.join(self.root, 'raw', file_name)
                data = np.load(npz_path)
                grasp_xyz = data['seg_center'].reshape(-1)
                grasp_rot = data['axes'].reshape(3,3)
                grasp_trans = np.zeros((1, 4, 4))
                grasp_trans[0, :3, :3] = grasp_rot
                grasp_trans[0, :3, 3] = grasp_xyz
                grasp_trans[0, 3, 3] = 1


        assert conditional_pc is not None
        for i in range(len(data_list)):
            data_list[i]['pc'] = torch.tensor(conditional_pc).unsqueeze(0).to(torch.float32)
            data_list[i]['grasp'] = torch.tensor(grasp_trans).to(torch.float32)

            # TODO: add grasp pose to data
        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)


    def process_riemanngrasp(self, cfg):
        data_list = []
        raw_files = self.raw_file_names

        grasp_file = [file for file in raw_files if 'riemann_center' in file]
        joint_file = [file for file in raw_files if 'txt' in file]

        # process grasp first
        riemann_path = os.path.join(self.root, 'raw', grasp_file[0])
        riemann_data = np.load(riemann_path)


        txt_path = os.path.join(self.root, 'raw', joint_file[0])
        jpos_mat = np.loadtxt(txt_path)  # contain 50 demos

        change_grasp_every = np.ceil(jpos_mat.shape[0] / riemann_data['xyz'].shape[0])
        change_grasp_id = 0
        cur_pc = None
        cur_grasp_pose = None
        for i in tqdm(range(jpos_mat.shape[0])): 

            line_rcd = torch.tensor(jpos_mat[i], dtype=torch.float32) # t, arm_left_6d, arm_right_6d
            joint_pose = line_rcd[1:].reshape(-1, cfg.dof).unsqueeze(0)

            data_slice = {'jpose': joint_pose}

            # update the grasp every change_grasp_every
            if i % change_grasp_every == 0:
                j = change_grasp_id
                obj_pc = riemann_data['xyz'][j].astype(np.float32)
                seg_center = riemann_data['seg_center'][j].astype(np.float32)
                axes = riemann_data['axes'][j].astype(np.float32)
                grasp_rot = axes.reshape(3, 3)
                obj_point = riemann_data['obj_point'][j].astype(np.float32)

                conditional_pc = obj_pc

                conditional_pc = downsample_pc(conditional_pc, cfg.num_points)

                cur_pc =  torch.tensor(conditional_pc).unsqueeze(0).to(torch.float32)

                # process grasp
                grasp_pose = np.zeros((4, 4)).astype(np.float32)
                grasp_pose[:3, :3] = grasp_rot
                grasp_pose[:3, 3] = seg_center
                grasp_pose[3, 3] = 1
                grasp_pose = grasp_pose.reshape(1, 4, 4)

                # cur_grasp_pose = self.trans2vec_pt3d(grasp_pose)
                cur_grasp_pose = grasp_pose

                change_grasp_id += 1

            data_slice['pc'] = cur_pc
            data_slice['grasp'] = cur_grasp_pose
            data_list.append(data_slice)


        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('######Loaded grasp data of length: ', len(data_list))



    def process_sam_predeff(self, cfg):
        print('Processing SAM dataset...')
        data_list = []
        raw_files = self.raw_file_names

        conditional_pc = None
        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]
            
            if 'hdf5' in  file_name:
                hdf5_path = os.path.join(self.root, 'raw', file_name)
                import h5py
                with h5py.File(hdf5_path, 'r') as f:

                    pred_pcs = f['pred_pcs'][()]
                    pc_num, pc_size, _ = pred_pcs.shape
                    assert pc_size == cfg.num_points
                    # start_offset = np.min(pred_pcs, axis=0)
                    # conditional_pc = pred_pcs - start_offset

                    end_pc = f['eff_pc'][()]
                    pc_size, _ = end_pc.shape
                    assert pc_size == cfg.num_points
                    end_offset = np.min(end_pc, axis=0)

                    pred_grasp_poses = f['pred_grasps'][()]
                    pred_grasp_num = pred_grasp_poses.shape[0]
                    eff_grasp_poses = f['eff_grasps'][()]
                    eff_grasp_num = eff_grasp_poses.shape[0]

                    joint_data = f['demo_joint_vals'][()]
                    stage = 'precondition'
                    selected_joint_data = []
                    for i in range(len(joint_data)):
                        left_jpose = joint_data[i][:6]
                        right_jpose = joint_data[i][7:13]
                        joint_pose = np.vstack((left_jpose, right_jpose)).reshape(1, -1, self.dof)


                        # only include jpose before OR after the action
                        stage = self.which_stage(stage, left_jpose, right_jpose)
                        if stage != cfg.tamp_type:
                            continue

                        selected_joint_data.append(joint_pose)

                    for pred_grasp_id in range(pred_grasp_num):
                        pred_pc = pred_pcs[pred_grasp_id].copy()
                        pred_offset = np.min(pred_pc, axis=0)
                        conditional_pc = pred_pc - pred_offset
                        pc_tensor = torch.tensor(conditional_pc).unsqueeze(0).to(torch.float32)

                        pred_grasp = pred_grasp_poses[pred_grasp_id].copy()
                        pred_grasp[:3, 3] -= pred_offset
                        pred_grasp_tensor = torch.tensor(pred_grasp).to(torch.float32).reshape(1, 4, 4)

                        eff_grasp_id = np.random.randint(0, eff_grasp_num)
                        #### substract the offset using center of the object
                        eff_grasp = eff_grasp_poses[eff_grasp_id].copy()
                        eff_grasp[:3, 3] -= end_offset
                        eff_grasp_tensor = torch.tensor(eff_grasp).to(torch.float32).reshape(1, 4, 4)

                        grasp_tensor = torch.cat((pred_grasp_tensor, eff_grasp_tensor), dim=1) # 1, 8, 4

                        joint_id = np.random.randint(0, len(selected_joint_data))
                        joint_pose = selected_joint_data[joint_id]

                        data = {'jpose': joint_pose, 'pc': pc_tensor, \
                                'grasp':grasp_tensor}
                        data_list.append(data)

        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 file!')

    # add eff grasp
    def process_50demos_predeff(self, cfg,  est_effpose = False):
        if self.has_eff and est_effpose:
            self.pretrained_encoder = VecDGCNN_att_frozen(preload_path= cfg.preload_path).cuda()    
        print('Processing hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names

        conditional_pc = None
        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]

            if 'hdf5' in  file_name:
                hdf5_path = os.path.join(self.root, 'raw', file_name)
                import h5py
                with h5py.File(hdf5_path, 'r') as f:

                    start_pc = f['start_grasps']['obj_points'][()]
                    pred_grasp_poses = f['start_grasps']['grasp_poses'][()]

                    if self.has_eff:
                        end_pc = f['end_grasps']['obj_points'][()]

                        if est_effpose:
                            # end_pc = rotate_around_z(end_pc, np.pi)
                            R_cuda, t_cuda = solve_pairwise_registration(self.pretrained_encoder, torch.tensor\
                                (start_pc).unsqueeze(0).float().cuda(), torch.tensor(end_pc).unsqueeze(0).float().cuda())
                            
                            # debug_and_save(start_pc, end_pc, R_cuda, t_cuda)
                        else:
                            end_offset = np.min(end_pc, axis=0)

                        eff_grasp_poses = f['end_grasps']['grasp_poses'][()]

                    joint_data = f['demo_joint_vals'][()]
                    stage = 'ungrasped'
                    for i in range(len(joint_data)):
                        left_jpose = joint_data[i][:self.dof]
                        right_jpose = joint_data[i][self.dof:]
                        joint_pose = np.vstack((left_jpose[:self.dof], right_jpose[:self.dof])).reshape(1, -1, self.dof)

                        # only include jpose before OR after the action
                        stage = self.which_stage(stage, left_jpose, right_jpose, threthold = 0.14, lifted_height = 0.12)
                        if stage != cfg.tamp_type:
                            continue

                        ## if obj_centric, cond_pc = raw_pc - offset; otherwise cond_pc = raw_pc
                        conditional_pc, start_offset = self.centralize_cond_pc(start_pc, self.is_obj_centric)

                        pc_tensor = torch.tensor(conditional_pc).unsqueeze(0).to(torch.float32)
                        pred_grasp_id = np.random.randint(0, len(pred_grasp_poses)-1)
                        pred_grasp = pred_grasp_poses[pred_grasp_id].copy()

                        if self.is_obj_centric:
                            pred_grasp[:3, 3] -= start_offset
                        grasp_tensor = torch.tensor(pred_grasp).to(torch.float32).reshape(1, 4, 4)
                        
                        if self.has_eff:
                            eff_grasp_id = np.random.randint(0, len(eff_grasp_poses)-1)
                            eff_grasp = eff_grasp_poses[eff_grasp_id].copy()
                            
                            if est_effpose:
                            ####  use ICP to estimate the rotation of the offset

                                R_cpu = R_cuda.squeeze().cpu().numpy()
                                t_cpu = t_cuda.squeeze().cpu().numpy()
                                transform_mat = np.zeros((4, 4))
                                transform_mat[:3, :3] = R_cpu
                                transform_mat[:3, 3] = t_cpu
                                transform_mat[3, 3] = 1
                                eff_grasp = np.dot(transform_mat, eff_grasp)
                            else:
                                #### substract the offset using center of the object
                                eff_grasp[:3, 3] -= end_offset
                                if not self.is_obj_centric:
                                    eff_grasp[:3, 3] += np.mean(start_pc, axis=0)
                            eff_grasp_tensor = torch.tensor(eff_grasp).to(torch.float32).reshape(1, 4, 4)
                            grasp_tensor = torch.cat((grasp_tensor, eff_grasp_tensor), dim=1) # 1, 8, 4
                            
                        data = {'jpose': joint_pose, 'pc': pc_tensor, \
                                'grasp':grasp_tensor}
                        data_list.append(data)
                    if stage != 'effect':
                        print(f'Warning: some data is not processed for {hdf5_path}')

        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 file!')




    # tell the stage from eef pose
    def which_stage(self, stage, left_jpose, right_jpose, threthold = 0.18, lifted_height = 0.07):
        left_arm_jpose = left_jpose[:self.dof]
        right_arm_jpose = right_jpose[:self.dof]
        left_gripper_val = left_jpose[-1]
        right_gripper_val = right_jpose[-1]
        #compute ee pose and see if they are too close
        eepose_l = qpos_to_eepose(left_arm_jpose, 0)
        eepose_r = qpos_to_eepose(right_arm_jpose, 1)
        eef_dist = np.linalg.norm(eepose_l[0] - eepose_r[0])

        if stage == 'ungrasped':
            if eepose_r[0][2] > lifted_height and right_gripper_val < 0.35:
                stage = 'precondition'
        elif stage == 'precondition':
            if eef_dist < threthold:
                stage = 'acting'
        elif stage == 'acting':
            if eef_dist > threthold:
                stage = 'effect'
        return stage
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    ## TODO: support more than 1 object
    def process_hdf5_mini(self, cfg, est_effpose = False):
        if self.has_eff and est_effpose:
            self.pretrained_encoder = VecDGCNN_att_frozen(preload_path= cfg.preload_path).cuda()    
        print('Processing hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names

        conditional_pc = None
        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]

            if 'hdf5' in  file_name:
                hdf5_path = os.path.join(self.root, 'raw', file_name)
                import h5py
                with h5py.File(hdf5_path, 'r') as f:
                    # list all keys
                    obj_names = list(f.keys())
                    for obj_name in obj_names:
                        start_pc = f[obj_name]['start_pc'][()]
                        pred_grasp_poses = f[obj_name]['grasp_poses'][()]
                        joint_data = f[obj_name]['joint_poses'][()]

                        if self.has_eff:
                            end_pc = f[obj_name]['end_pc'][()]
                            if est_effpose:
                                # end_pc = rotate_around_z(end_pc, np.pi)
                                R_cuda, t_cuda = solve_pairwise_registration(self.pretrained_encoder, torch.tensor\
                                    (start_pc).unsqueeze(0).float().cuda(), torch.tensor(end_pc).unsqueeze(0).float().cuda())
                                debug_and_save(start_pc, end_pc, R_cuda, t_cuda)
                            else:
                                end_offset = np.min(end_pc, axis=0)

                            eff_grasp_poses = f[obj_name]['release_poses'][()]

                        assert len(joint_data) > len(pred_grasp_poses)
                        
                    for i in range(len(joint_data)):
                        assert len(joint_data[i]) == 2*self.dof
                        left_jpose = joint_data[i][:self.dof]
                        right_jpose = joint_data[i][self.dof:]
                        joint_pose = np.vstack((left_jpose, right_jpose)).reshape(1, -1, self.dof)

                        ## if obj_centric, cond_pc = raw_pc - offset; otherwise cond_pc = raw_pc
                        conditional_pc, start_offset = self.centralize_cond_pc(start_pc, self.is_obj_centric)

                        pc_tensor = torch.tensor(conditional_pc).unsqueeze(0).to(torch.float32)
                        pred_grasp_id = np.random.randint(0, len(pred_grasp_poses)-1)
                        pred_grasp = pred_grasp_poses[pred_grasp_id].copy()

                        if self.is_obj_centric:
                            pred_grasp[:3, 3] -= start_offset
                        grasp_tensor = torch.tensor(pred_grasp).to(torch.float32).reshape(1, 4, 4)
                        
                        if self.has_eff:
                            eff_grasp_id = np.random.randint(0, len(eff_grasp_poses)-1)
                            eff_grasp = eff_grasp_poses[eff_grasp_id].copy()
                            
                            if est_effpose:
                            ####  use ICP to estimate the rotation of the offset

                                R_cpu = R_cuda.squeeze().cpu().numpy()
                                t_cpu = t_cuda.squeeze().cpu().numpy()
                                transform_mat = np.zeros((4, 4))
                                transform_mat[:3, :3] = R_cpu
                                transform_mat[:3, 3] = t_cpu
                                transform_mat[3, 3] = 1
                                eff_grasp = np.dot(transform_mat, eff_grasp)
                            else:
                                #### substract the offset using center of the object
                                eff_grasp[:3, 3] -= end_offset
                                if not self.is_obj_centric:
                                    eff_grasp[:3, 3] += np.mean(start_pc, axis=0)
                            eff_grasp_tensor = torch.tensor(eff_grasp).to(torch.float32).reshape(1, 4, 4)

                            grasp_tensor = torch.cat((grasp_tensor, eff_grasp_tensor), dim=1) # 1, 8, 4
                        data = {'jpose': joint_pose, 'pc': pc_tensor, \
                                'grasp':grasp_tensor}
                        data_list.append(data)

        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 file!')


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

@hydra.main(config_path=os.path.join(EQUIBOT_PATH, "equibot/policies/configs"), config_name="transfer_tape")
def main(cfg):
    cfg.data.dataset.path=os.path.join(EQUIBOT_PATH, 'data/transfer_cup/')
    test_dataset = ALOHAPoseDataset(cfg.data.dataset, "test", est_effpose = False, force_process=True)
    num_workers = 0
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    
    for batch_id, batch in enumerate(test_loader):

        # rot_list = [0, np.pi/2, np.pi, np.pi/2*3]
        rot_list = [0]
        for rot_z in rot_list:
            np_obs= rotate_observation(batch, rot_z)
            cpu_obs = to_tensor(np_obs)

            history_list = []
            tmp_pc = cpu_obs['pc'][0].reshape(-1, 3).numpy()
            for i in range(cpu_obs['jpose'].shape[0]):
                jpose = cpu_obs['jpose'][i].reshape(-1).numpy()
                grasp_pose = cpu_obs['grasp'][i].reshape(-1,4).numpy()
                grasp_pose = grasp_pose[:4]

                ## ## To validate that it is equivalent to rotate the grasp vector(rotate_vec_grasp) and rotate the transformation matrix(rotate_observation)
                origin_pred_grasp = batch['grasp'].detach().cpu()[i][:, :4, :].reshape(1, 1, 4, 4)
                ref_pred_grasp = cpu_obs['grasp'].detach().cpu()[i][:, :4, :].reshape(1, 1, 4, 4)
                vecrot_grasp = rotate_vec_grasp(origin_pred_grasp, rot_z)
                trans_error = torch.norm(vecrot_grasp - ref_pred_grasp)
                print('the error of two computed grasp is: ',trans_error) 

                # action_slice = (grasp, jpose)
                action_slice = (vecrot_grasp.reshape(-1, 4), jpose)
                history_list.append(action_slice)

            render_pose(history_list, use_gui=True, \
                        directory = None, obj_points = tmp_pc)



if __name__ == '__main__':
    main()