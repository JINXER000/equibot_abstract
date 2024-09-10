import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import namedtuple
from equibot.policies.utils.constants import qpos_to_eepose

import hydra
import sys
sys.path.append('/home/user/yzchen_ws/TAMP-ubuntu22/pddlstream_aloha')
sys.path.append('/mnt/TAMP/interbotix_ws/src/pddlstream_aloha')
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose
from examples.pybullet.aloha_real.scripts.constants import qpos_to_eepose


feature_tuple = namedtuple('feature_tuple', ['dim', 'start', 'end'])

class ALOHAPoseDataset(Dataset):
    def __init__(self, cfg, mode, transform=None, pre_transform=None, pre_filter=None):
        super().__init__()
        self.mode = mode
        self.dir_name = cfg.path
        self.root = self.dir_name
        self.symb_mask = cfg.symb_mask
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.composed_inference = False

        
        # Process the data
        self.process_select(cfg)

        # if not os.path.exists(self.processed_file_path):
        #     print('NOTE: dataset already processed!')
        #     self.process_select(cfg)
        
        # Load processed data
        self.data, self.slices = torch.load(self.processed_file_path)

    def process_select(self, cfg):
        if cfg.dataset_type == 'hdf5':
            self.process_50demos(cfg)
        elif cfg.dataset_type == 'npz':
            self.process_riemanngrasp(cfg)
        elif cfg.dataset_type == 'txt':
            self.process_txt(cfg)
        elif cfg.dataset_type == 'one_hdf5':
            self.process_one_hdf5(cfg)
        elif cfg.dataset_type == 'hdf5_predeff':
            self.process_50demos_predeff(cfg)
        else:
            raise NotImplementedError('Dataset type not implemented!')


    @property
    def raw_file_names(self):
        return os.listdir(os.path.join(self.root, 'raw'))

    @property
    def processed_file_path(self):
        return os.path.join(self.root, 'processed', 'data.pt')

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
                    # data = {'demo_t': demo_t, 'joint_pose': joint_pose, 'mask': mask_data}
                    data = {'demo_t': demo_t, 'joint_pose': joint_pose}
                    data_list.append(data)
            elif  file_name  == 'graspobj_4.ply':
                ply_path = os.path.join(self.root, 'raw', file_name)
                import open3d as o3d
                conditional_pc = o3d.io.read_point_cloud(ply_path)
                conditional_pc = np.asarray(conditional_pc.points)

                tgt_size = cfg.num_points
                sampled_indices = np.random.choice(conditional_pc.shape[0], tgt_size, replace=False)
                conditional_pc = conditional_pc[sampled_indices]

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
            data_list[i]['grasp_pose'] = torch.tensor(grasp_trans).to(torch.float32)

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

                    conditional_pc = f['start_grasps']['obj_points'][()]
                    tgt_size = cfg.num_points
                    sampled_indices = np.random.choice(conditional_pc.shape[0], tgt_size, replace=False)
                    conditional_pc = conditional_pc[sampled_indices]

                    if cfg.tamp_type == 'precondition':
                        grasp_poses = f['start_grasps']['grasp_poses'][()]
                    elif cfg.tamp_type == 'effect':
                        grasp_poses = f['end_grasps']['grasp_poses'][()]

                    grasp_nums = len(grasp_poses)
                    joint_data = f['demo_joint_vals'][()]
                    stage = 'precondition'
                    for i in range(len(joint_data)):
                        left_jpose = joint_data[i][:6]
                        right_jpose = joint_data[i][7:13]

                        # only include jpose before OR after the action
                        stage = self.which_stage(stage, left_jpose, right_jpose)
                        if stage != cfg.tamp_type:
                            continue

                        if self.symb_mask[0] == 'None':
                            joint_pose = right_jpose.reshape(1, 1, 6)
                        elif self.symb_mask[1] == 'None':
                            joint_pose = left_jpose.reshape(1, 1, 6)
                        else: # num_eef ==2
                            joint_pose = np.concatenate((left_jpose, right_jpose)).reshape(1, 2, 6)

                        grasp_id = np.random.randint(0, grasp_nums-1)
                        pc_tensor = torch.tensor(conditional_pc).unsqueeze(0).to(torch.float32)
                        grasp_tensor = torch.tensor(grasp_poses[grasp_id]).to(torch.float32).reshape(1, 4, 4)
                        data = {'joint_pose': joint_pose, 'pc': pc_tensor, \
                                'grasp_pose':grasp_tensor}
                        data_list.append(data)

        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 file!')

    # add eff grasp
    def process_50demos_predeff(self, cfg):
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
                    tgt_size = cfg.num_points
                    sampled_indices = np.random.choice(start_pc.shape[0], tgt_size, replace=False)
                    start_pc = start_pc[sampled_indices]
                    start_offset = np.min(start_pc, axis=0)
                    conditional_pc = start_pc - start_offset

                    end_pc = f['end_grasps']['obj_points'][()]
                    sampled_indices = np.random.choice(end_pc.shape[0], tgt_size, replace=False)
                    end_pc = end_pc[sampled_indices]
                    end_offset = np.min(end_pc, axis=0)
                    # offset_diff = start_offset - end_offset

                    pred_grasp_poses = f['start_grasps']['grasp_poses'][()]
                    eff_grasp_poses = f['end_grasps']['grasp_poses'][()]

                    joint_data = f['demo_joint_vals'][()]
                    stage = 'precondition'
                    for i in range(len(joint_data)):
                        left_jpose = joint_data[i][:6]
                        right_jpose = joint_data[i][7:13]

                        # only include jpose before OR after the action
                        stage = self.which_stage(stage, left_jpose, right_jpose)
                        if stage != cfg.tamp_type:
                            continue

                        if self.symb_mask[0] == 'None':
                            joint_pose = right_jpose.reshape(1, 1, 6)
                        elif self.symb_mask[1] == 'None':
                            joint_pose = left_jpose.reshape(1, 1, 6)
                        else: # num_eef ==2
                            joint_pose = np.concatenate((left_jpose, right_jpose)).reshape(1, 2, 6)

                        pc_tensor = torch.tensor(conditional_pc).unsqueeze(0).to(torch.float32)
                        pred_grasp_id = np.random.randint(0, len(pred_grasp_poses)-1)
                        pred_grasp = pred_grasp_poses[pred_grasp_id].copy()
                        pred_grasp[:3, 3] -= start_offset
                        pred_grasp_tensor = torch.tensor(pred_grasp).to(torch.float32).reshape(1, 4, 4)
                        
                        eff_grasp_id = np.random.randint(0, len(eff_grasp_poses)-1)
                        #### substract the offset using center of the object
                        #### TODO: use ICP to estimate the rotation of the offset
                        eff_grasp = eff_grasp_poses[eff_grasp_id].copy()
                        eff_grasp[:3, 3] -= end_offset
                        eff_grasp_tensor = torch.tensor(eff_grasp).to(torch.float32).reshape(1, 4, 4)

                        grasp_tensor = torch.cat((pred_grasp_tensor, eff_grasp_tensor), dim=1) # 1, 8, 4
                        data = {'joint_pose': joint_pose, 'pc': pc_tensor, \
                                'grasp_pose':grasp_tensor}
                        data_list.append(data)

        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 file!')


    # tell the stage from eef pose
    def which_stage(self, stage, left_jpose, right_jpose, threthold = 0.18):
        #compute ee pose and see if they are too close
        eepose_l = qpos_to_eepose(left_jpose, 0)
        eepose_r = qpos_to_eepose(right_jpose, 1)
        eef_dist = np.linalg.norm(eepose_l[0] - eepose_r[0])

        if stage == 'precondition':
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

@hydra.main(config_path="/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/equibot/policies/configs", config_name="transfer_tape")
def main(cfg):
    cfg.data.dataset.path='/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/data/transfer_tape/'
    test_dataset = ALOHAPoseDataset(cfg.data.dataset, "test")
    num_workers = cfg.data.dataset.num_workers
    batch_size = 32
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    
    for batch_id, batch in enumerate(test_loader):
        history_list = []
        tmp_pc = batch['pc'][0].reshape(-1, 3).numpy()
        for i in range(batch['joint_pose'].shape[0]):
            jpose = batch['joint_pose'][i].reshape(-1).numpy()
            grasp_pose = batch['grasp_pose'][i].reshape(-1,4).numpy()
            action_slice = (grasp_pose, jpose)
            # action_slice = (None, jpose)
            history_list.append(action_slice)


        render_pose(history_list, use_gui=True, \
                    directory = None, obj_points = tmp_pc)

if __name__ == '__main__':
    main()