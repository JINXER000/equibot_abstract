import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import namedtuple

DATASET_PATH = '/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/data/insert_tape'
feature_tuple = namedtuple('feature_tuple', ['dim', 'start', 'end'])


class ALOHAPoseDataset(Dataset):
    # def __init__(self, dir_name, transform=None, pre_transform=None, pre_filter=None, symb_mask=['qpose_left', 'qpose_right', None, None, None]):
    #     self.dir_name = dir_name
    #     self.root = os.path.join(DATASET_PATH, dir_name)
    #     self.transform = transform
    #     self.pre_transform = pre_transform
    #     self.pre_filter = pre_filter
    #     self.symb_mask = symb_mask

    def __init__(self, cfg, mode, transform=None, pre_transform=None, pre_filter=None):
        super().__init__()
        self.mode = mode
        self.dir_name = cfg.path
        self.root = os.path.join(DATASET_PATH, self.dir_name)
        self.symb_mask = cfg.symb_mask
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.composed_inference = False

        # # Calculate mask
        # self.dims = self.calc_dims(self.symb_mask)
        # # mask = []
        # # for i in range(len(self.symb_mask)):
        # #     if self.symb_mask[i] is None:
        # #         mask += [True for _ in range(self.dims[i].dim)]
        # #     else:
        # #         mask += [False for _ in range(self.dims[i].dim)]
        
        # # self.mask = torch.tensor(mask, dtype=torch.bool)
        
        # Process the data
        # self.process(cfg)
        self.process_hdf5(cfg)
        # if not os.path.exists(self.processed_file_path):
        #     self.process(cfg)
        
        # Load processed data
        self.data, self.slices = torch.load(self.processed_file_path)

    # def calc_dims(self, symb_mask):
    #     grasp_dim = 19
    #     qpose_dim = 6
    #     objpose_dim = 5
    #     dims = []
    #     start_idx = 0
    #     # qpose
    #     for i in range(2):
    #         dims.append(feature_tuple(qpose_dim, start_idx, qpose_dim + start_idx))
    #         start_idx += qpose_dim
    #     # grasp
    #     for i in range(2, 4):
    #         dims.append(feature_tuple(grasp_dim, start_idx, grasp_dim + start_idx))
    #         start_idx += grasp_dim
    #     # objpose
    #     for i in range(4, len(symb_mask)):
    #         dims.append(feature_tuple(objpose_dim, start_idx, objpose_dim + start_idx))
    #         start_idx += objpose_dim

    #     dims = tuple(dims)
    #     return dims

    @property
    def raw_file_names(self):
        return os.listdir(os.path.join(self.root, 'raw'))

    @property
    def processed_file_path(self):
        return os.path.join(self.root, 'processed', 'data.pt')

    def process(self, cfg):
        print('Processing dataset...')
        data_list = []
        raw_files = self.raw_file_names

        conditional_pc = None
        grasp_xyz = None
        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]
            # joint pose
            if 'jpose' in file_name:
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
                    # data = {'demo_t': demo_t, 'joint_pose': joint_pose, 'mask': mask_data}
                    data = {'demo_t': demo_t, 'joint_pose': joint_pose}
                    data_list.append(data)
            elif 'ply' in file_name:
                ply_path = os.path.join(self.root, 'raw', file_name)
                import open3d as o3d
                conditional_pc = o3d.io.read_point_cloud(ply_path)
                conditional_pc = np.asarray(conditional_pc.points)

                tgt_size = cfg.num_points
                sampled_indices = np.random.choice(conditional_pc.shape[0], tgt_size, replace=False)
                conditional_pc = conditional_pc[sampled_indices]

            elif 'npz' in file_name: # as a dummy input of vnn
                npz_path = os.path.join(self.root, 'raw', file_name)
                data = np.load(npz_path)
                grasp_xyz = data['seg_center'].reshape(1,1,3)

        assert conditional_pc is not None
        for i in range(len(data_list)):
            data_list[i]['pc'] = torch.tensor(conditional_pc).unsqueeze(0).to(torch.float32)
            data_list[i]['grasp_xyz'] = torch.tensor(grasp_xyz).to(torch.float32)

            # TODO: add grasp pose to data
        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)

    def process_hdf5(self, cfg):
        print('Processing hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names

        conditional_pc = None
        grasp_xyz = None
        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]
            
            if 'hdf5' in file_name:
                hdf5_path = os.path.join(self.root, 'raw', file_name)
                import h5py
                with h5py.File(hdf5_path, 'r') as f:
                    joint_data = f['demo_joint_vals'][()]
                    for i in range(len(joint_data)):
                        left_jpose = joint_data[i][:6]
                        right_jpose = joint_data[i][7:13]
                        joint_pose = np.concatenate((left_jpose, right_jpose)).reshape(1, 2, 6)

                        data = {'joint_pose': joint_pose}
                        data_list.append(data)

                    if cfg.tamp_type == 'precondition':
                        conditional_pc = f['start_grasps']['obj_points'][()]
                    elif cfg.tamp_type == 'effect':
                        conditional_pc = f['end_grasps']['obj_points'][()]
                    tgt_size = cfg.num_points
                    sampled_indices = np.random.choice(conditional_pc.shape[0], tgt_size, replace=False)
                    conditional_pc = conditional_pc[sampled_indices]

                    if cfg.tamp_type == 'precondition':
                        grasp_poses = f['start_grasps']['grasp_poses'][()]
                    elif cfg.tamp_type == 'effect':
                        grasp_poses = f['end_grasps']['grasp_poses'][()]

                    

        for i in range(len(data_list)):
            data_list[i]['pc'] = torch.tensor(conditional_pc).unsqueeze(0).to(torch.float32)

            # select the 1st grasp conditioned on the pc.
            grasp_vec = self.trans2vec(grasp_poses)[0]
            data_list[i]['grasp_pose'] = torch.tensor(grasp_vec).to(torch.float32)

            # TODO: add grasp pose to data
        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)

    # Q: should grasp be 3x3 or 1x9? I think both ok, because at last the dim will be (-1, 3)
    def trans2vec(self, grasp_trans_arr):

        grasp_xyz = grasp_trans_arr[:, :3, 3].reshape(-1, 3)
        eef_rot = grasp_trans_arr[:, :3, :3]
        dir1 = eef_rot[:, :3, 0]
        dir2 = eef_rot[:, :3, 2]
        for i in range(len(dir1)):
            assert np.allclose(
                np.cross(dir2[i], dir1[i]), eef_rot[:, :, 1][i], atol=1e-4
            )        
        grasp = np.concatenate((grasp_xyz, dir1, dir2), axis=1)
        grasp = grasp.reshape(-1, 1, 1, 9)
        return grasp



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
