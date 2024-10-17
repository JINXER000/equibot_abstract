from .abstract_dataset import ALOHAPoseDataset
import os
import numpy as np
import torch


class DualAbsDataset(ALOHAPoseDataset):
    def __init__(self, cfg, mode, transform=None):
        super(DualAbsDataset, self).__init__(cfg, mode, transform)

    def process_select(self, cfg):
        self.process_mj_insertion_pred(cfg)

        # if cfg.dataset_type == 'mj_insertion_pred':
        #     self.process_mj_insertion_pred(cfg)
        # elif cfg.dataset_type == 'mj_socket_test':
        #     self.process_mj_socket_test(cfg)
        # else:
        #     raise NotImplementedError('Dataset type not implemented!')
        

    ## TODO: need to postprocess to get the grasp pose
    def process_mj_insertion_pred(self, cfg):
        print('Processing mj hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names

        
        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]
            
            if 'hdf5' in  file_name:
                hdf5_path = os.path.join(self.root, 'raw', file_name)
                import h5py
                with h5py.File(hdf5_path, 'r') as f:

                    socket_pc = f['socket_grasps']['obj_points'][()]
                    socket_pc_n, socket_offset = self.centralize_cond_pc( socket_pc)
                    socket_pc_tensor = torch.tensor(socket_pc_n).unsqueeze(0).\
                        to(torch.float32).reshape(1, cfg.num_points, 3)

                    peg_pc = f['peg_grasps']['obj_points'][()]
                    peg_pc_n, peg_offset = self.centralize_cond_pc( peg_pc)
                    peg_pc_tensor = torch.tensor(peg_pc_n).unsqueeze(0).\
                        to(torch.float32).reshape(1, cfg.num_points, 3)


                    ### process grasp and joint pose

                    socket_grasp_poses = f['socket_grasps']['grasp_poses'][()]
                    peg_grasp_poses = f['peg_grasps']['grasp_poses'][()]

                    joint_data = f['pred_joint_vals'][()]
                    stage = 'precondition'
                    for i in range(len(joint_data)):
                        left_jpose = joint_data[i][:6]
                        right_jpose = joint_data[i][7:13]

                        # only include jpose before OR after the action
                        stage = self.which_stage(stage, left_jpose, right_jpose)
                        if stage != cfg.tamp_type:
                            continue

                        ## add gripper action (claw)
                        left_jpose = np.concatenate((left_jpose, np.array([joint_data[i][7]])))
                        left_jpose_tensor = torch.tensor(left_jpose).to(torch.float32).reshape(1, 1, -1)
                        right_jpose = np.concatenate((right_jpose, np.array([joint_data[i][-1]])))
                        right_jpose_tensor = torch.tensor(right_jpose).to(torch.float32).reshape(1, 1, -1)
                        
                        grasp_id = np.random.randint(0, len(socket_grasp_poses)-1)

                        socket_grasp = socket_grasp_poses[grasp_id].copy()
                        socket_grasp = self.centralize_grasp(socket_grasp, socket_offset)
                        socket_grasp_tensor = torch.tensor(socket_grasp).to(torch.float32).reshape(1, 4, 4)
                        
                        peg_grasp = peg_grasp_poses[grasp_id].copy()
                        peg_grasp = self.centralize_grasp(peg_grasp, peg_offset)
                        peg_grasp_tensor = torch.tensor(peg_grasp).to(torch.float32).reshape(1, 4, 4)

                        if "socket" in cfg.dataset_type:
                            data = {'jpose': left_jpose_tensor, 
                                    'pc': socket_pc_tensor, 
                                    'grasp': socket_grasp_tensor}
                        elif "peg" in cfg.dataset_type:
                            data = {'jpose': right_jpose_tensor, 
                                    'pc': peg_pc_tensor, 
                                    'grasp': peg_grasp_tensor}
                        else:
                            data = {'left_jpose': left_jpose_tensor, 'right_jpose': right_jpose_tensor,\
                                    'left_pc': socket_pc_tensor, 'right_pc': peg_pc_tensor,\
                                    'left_grasp': socket_grasp_tensor, 'right_grasp': peg_grasp_tensor}
                        data_list.append(data)

        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 file!')



    # def process_mj_socket_test(self, cfg):
    #     print('Processing socket mj hdf5 dataset...')
    #     data_list = []
    #     raw_files = self.raw_file_names

    #     mj_offset = np.array([0.0, -0.5, 0.0])
    #     for file_id in range(len(raw_files)):
    #         file_name = raw_files[file_id]
            
    #         if 'hdf5' in  file_name:
    #             hdf5_path = os.path.join(self.root, 'raw', file_name)
    #             import h5py
    #             with h5py.File(hdf5_path, 'r') as f:

    #                 socket_pc = f['socket_grasps']['obj_points'][()]
    #                 assert socket_pc.shape[0] == cfg.num_points
    #                 # sampled_indices = np.random.choice(socket_pc.shape[0], tgt_size, replace=False)
    #                 # socket_pc = socket_pc[sampled_indices]
    #                 socket_offset = np.min(socket_pc, axis=0) + mj_offset
    #                 socket_pc_tensor = torch.tensor(socket_pc - socket_offset).unsqueeze(0).\
    #                     to(torch.float32).reshape(1, cfg.num_points, 3)

    #                 # peg_pc = f['peg_grasps']['obj_points'][()]
    #                 # assert peg_pc.shape[0] == cfg.num_points
    #                 # # sampled_indices = np.random.choice(peg_pc.shape[0], tgt_size, replace=False)
    #                 # # peg_pc = peg_pc[sampled_indices]
    #                 # peg_offset = np.min(peg_pc, axis=0)
    #                 # peg_pc_tensor = torch.tensor(peg_pc - peg_offset).unsqueeze(0).\
    #                 #     to(torch.float32).reshape(1, cfg.num_points, 3)


    #                 ### process grasp and joint pose

    #                 socket_grasp_poses = f['socket_grasps']['grasp_poses'][()]
    #                 # peg_grasp_poses = f['peg_grasps']['grasp_poses'][()]

    #                 joint_data = f['pred_joint_vals'][()]
    #                 stage = 'precondition'
    #                 for i in range(len(joint_data)):
    #                     left_jpose = joint_data[i][:6]
    #                     right_jpose = joint_data[i][7:13]

    #                     # only include jpose before OR after the action
    #                     stage = self.which_stage(stage, left_jpose, right_jpose)
    #                     if stage != cfg.tamp_type:
    #                         continue

    #                     ## add gripper action (claw)
    #                     left_jpose = np.concatenate((left_jpose, np.array([joint_data[i][7]])))
    #                     left_jpose_tensor = torch.tensor(left_jpose).to(torch.float32).reshape(1, 1, -1)
    #                     # right_jpose = np.concatenate((right_jpose, np.array([joint_data[i][-1]])))
    #                     # right_jpose_tensor = torch.tensor(right_jpose).to(torch.float32).reshape(1, 1, -1)
                        
    #                     socket_grasp_id = np.random.randint(0, len(socket_grasp_poses)-1)
    #                     socket_grasp = socket_grasp_poses[socket_grasp_id].copy()
    #                     socket_grasp[:3, 3] -= socket_offset
    #                     socket_grasp_tensor = torch.tensor(socket_grasp).to(torch.float32).reshape(1, 4, 4)
                        
    #                     # peg_grasp_id = np.random.randint(0, len(peg_grasp_poses)-1)
    #                     # peg_grasp = peg_grasp_poses[peg_grasp_id].copy()
    #                     # peg_grasp[:3, 3] -= peg_offset
    #                     # peg_grasp_tensor = torch.tensor(peg_grasp).to(torch.float32).reshape(1, 4, 4)

    #                     # grasp_tensor = torch.cat((socket_grasp_tensor, peg_grasp_tensor), dim=1) # 1, 8, 4
    #                     # data = {'left_jpose': left_jpose_tensor, 'right_jpose': right_jpose_tensor,\
    #                     #         'left_pc': socket_pc_tensor, 'right_pc': peg_pc_tensor,\
    #                     #         'left_grasp': socket_grasp_tensor, 'right_grasp': peg_grasp_tensor}
    #                     data = {'jpose': left_jpose_tensor, 
    #                             'pc': socket_pc_tensor, 
    #                             'grasp': socket_grasp_tensor}

    #                     data_list.append(data)

    #     os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
    #     torch.save((data_list, None), self.processed_file_path)
    #     print('processed all hdf5 file!')

