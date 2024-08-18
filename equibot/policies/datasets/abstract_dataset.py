import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import namedtuple
from equibot.policies.utils.misc import  matrix_to_rotation_6d

# import pytorch3d as pt

DATASET_PATH = '/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/data/transfer_tape'
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
        self.root = self.dir_name
        # self.root = os.path.join(DATASET_PATH, self.dir_name)
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
        if cfg.dataset_type == 'hdf5':
            self.process_50demos(cfg)
            # self.process_one_hdf5(cfg)
        elif cfg.dataset_type == 'npz':
            self.process_riemanngrasp(cfg)
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

    def process_one_hdf5(self, cfg):
        print('Processing hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names

        conditional_pc = None
        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]
            
            if file_name == 'grasp_episode_99.hdf5':
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

            # data_list[i]['grasp_pose'] = self.trans2vec_pt3d(grasp_poses)
            data_list[i]['grasp_pose'] = torch.tensor(grasp_poses[0]).to(torch.float32).reshape(1, 4, 4)
           
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

            data_slice = {'joint_pose': joint_pose}

            # update the grasp every change_grasp_every
            if i % change_grasp_every == 0:
                j = change_grasp_id
                obj_pc = riemann_data['xyz'][j].astype(np.float32)
                seg_center = riemann_data['seg_center'][j].astype(np.float32)
                axes = riemann_data['axes'][j].astype(np.float32)
                grasp_rot = axes.reshape(3, 3)
                obj_point = riemann_data['obj_point'][j].astype(np.float32)

                conditional_pc = obj_pc

                tgt_size = cfg.num_points
                sampled_indices = np.random.choice(conditional_pc.shape[0], tgt_size, replace=False)
                conditional_pc = conditional_pc[sampled_indices]

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
            data_slice['grasp_pose'] = cur_grasp_pose
            data_list.append(data_slice)


        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('######Loaded grasp data of length: ', len(data_list))

    def process_50demos(self, cfg):
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

                    grasp_nums = len(grasp_poses)
                    joint_data = f['demo_joint_vals'][()]
                    for i in range(len(joint_data)):
                        left_jpose = joint_data[i][:6]
                        right_jpose = joint_data[i][7:13]
                        joint_pose = np.concatenate((left_jpose, right_jpose)).reshape(1, 2, 6)

                        grasp_id = np.random.randint(0, grasp_nums-1)
                        pc_tensort = torch.tensor(conditional_pc).unsqueeze(0).to(torch.float32)
                        grasp_tensor = torch.tensor(grasp_poses[grasp_id]).to(torch.float32).reshape(1, 4, 4)
                        data = {'joint_pose': joint_pose, 'pc': pc_tensort, \
                                'grasp_pose':grasp_tensor}
                        data_list.append(data)

        # change_grasp_every = np.ceil(len(data_list) / len(grasp_poses))
        # change_grasp_id = 0
        # for i in range(len(data_list)):
        #     data_list[i]['pc'] = torch.tensor(conditional_pc).unsqueeze(0).to(torch.float32)

        #     data_list[i]['grasp_pose'] = torch.tensor(grasp_poses[change_grasp_id]).to(torch.float32).reshape(1, 4, 4)

        #     # update the grasp every change_grasp_every
        #     if i % change_grasp_every == 0:
        #         change_grasp_id +=1

        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 file!')




    # def trans2vec_pt3d(self, grasp_trans_arr):
    #     grasp_trans_arr = torch.tensor(grasp_trans_arr).to(torch.float32)
    #     grasp_xyz = grasp_trans_arr[:, :3, 3].reshape(-1, 3)
    #     eef_rot = grasp_trans_arr[:, :3, :3]
    #     # map to cont space using pytorch 3d
    #     rot6d = matrix_to_rotation_6d(eef_rot)  # B*6
    #     grasp = torch.cat((grasp_xyz, rot6d), dim=1)  #  B*9

    #     grasp = grasp.reshape(1, 9)
    #     return grasp


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
