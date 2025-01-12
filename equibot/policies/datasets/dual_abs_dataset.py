from .abstract_dataset import ALOHAPoseDataset, solve_pairwise_registration, VecDGCNN_att_frozen
import os
import numpy as np
import torch


class DualAbsDataset(ALOHAPoseDataset):
    def __init__(self, cfg, mode, transform=None):
        self.has_eff_dict = {'left': False, 'right': False}
        hand_sides = ['left', 'right']
        for i in range(len(hand_sides)):
            self.has_eff_dict[hand_sides[i]] = cfg.has_eff_list[i]
            
        super(DualAbsDataset, self).__init__(cfg, mode, transform)

    def process_select(self, cfg):
        # print('saving time when debug!')
        # return
        if cfg.dataset_type == 'mj_insertion_pred':
            self.process_mj_insertion_pred(cfg)
        elif cfg.dataset_type == 'dual_hdf5_mini':
            self.process_dual_hdf5_mini(cfg)
        

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
                    peg_pc = f['peg_grasps']['obj_points'][()]

                    ### process grasp and joint pose

                    socket_grasp_poses = f['socket_grasps']['grasp_poses'][()]
                    peg_grasp_poses = f['peg_grasps']['grasp_poses'][()]

                    joint_data = f['pred_joint_vals'][()]
                    stage = 'precondition'
                    for i in range(len(joint_data)):
                        left_jpose = joint_data[i][:6]
                        right_jpose = joint_data[i][7:13]

                        # only include jpose before OR after the action
                        stage = self.which_stage(stage, left_jpose, right_jpose, lifted_height = 0.1)
                        if stage != cfg.tamp_type:
                            continue

                        ## downsample and centralize pc
                        socket_pc_n, socket_offset = self.centralize_cond_pc( socket_pc)
                        socket_pc_tensor = torch.tensor(socket_pc_n).unsqueeze(0).\
                            to(torch.float32).reshape(1, cfg.num_points, 3)

                        peg_pc_n, peg_offset = self.centralize_cond_pc( peg_pc)
                        peg_pc_tensor = torch.tensor(peg_pc_n).unsqueeze(0).\
                            to(torch.float32).reshape(1, cfg.num_points, 3)

                        ## add gripper action (claw)
                        left_jpose = np.concatenate((left_jpose, np.array([joint_data[i][7]])))
                        left_jpose_tensor = torch.tensor(left_jpose).to(torch.float32).reshape(1, 1, -1)
                        right_jpose = np.concatenate((right_jpose, np.array([joint_data[i][-1]])))
                        right_jpose_tensor = torch.tensor(right_jpose).to(torch.float32).reshape(1, 1, -1)
                        dual_jpose_tensor = torch.cat((left_jpose_tensor, right_jpose_tensor), dim=1) # 1, 2, 7
                        
                        socket_grasp_id = np.random.randint(0, len(socket_grasp_poses)-1)
                        socket_grasp = socket_grasp_poses[socket_grasp_id].copy()
                        socket_grasp = self.centralize_grasp(socket_grasp, socket_offset)
                        socket_grasp_tensor = torch.tensor(socket_grasp).to(torch.float32).reshape(1, 4, 4)
                        
                        peg_grasp_id = np.random.randint(0, len(peg_grasp_poses)-1)
                        peg_grasp = peg_grasp_poses[peg_grasp_id].copy()
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
                                    'left_grasp': socket_grasp_tensor, 'right_grasp': peg_grasp_tensor,
                                    'dual_jpose': dual_jpose_tensor}
                        data_list.append(data)

        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 file!')

    ## TODO: deal with jpose and grasp pose separately
    def process_dual_hdf5_mini(self, cfg,  est_effpose = False):

        has_eff = (True in cfg.has_eff_list)
        if has_eff and est_effpose:
            self.pretrained_encoder = VecDGCNN_att_frozen(preload_path= cfg.preload_path).cuda()    
        print('Processing hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names

        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]

            if 'hdf5' in  file_name:
                hdf5_path = os.path.join(self.root, 'raw', file_name)
                import h5py
                with h5py.File(hdf5_path, 'r') as f:

                    tensor_dict = {'left_grasp': [], 'right_grasp': [],\
                                   'left_pc': [], 'right_pc': [],\
                                   'dual_jpose': []}
                    # list all keys
                    dual_jpose_data = []
                    sides = list(f.keys())
                    for side in sides:
                        start_pc = f[side]['start_pc'][()]
                        pred_grasp_poses = f[side]['grasp_poses'][()]
                        ## NOTE: joint data in each object should be the same! e.g., 14 dof for aloha
                        joint_data_tmp = f[side]['joint_poses'][()]
                        if len(joint_data_tmp) != 0:
                            dual_jpose_data = joint_data_tmp   
                        if self.has_eff_dict[side]:
                            end_pc = f[side]['end_pc'][()]
                            if est_effpose:
                                # end_pc = rotate_around_z(end_pc, np.pi)
                                R_cuda, t_cuda = solve_pairwise_registration(self.pretrained_encoder, torch.tensor\
                                    (start_pc).unsqueeze(0).float().cuda(), torch.tensor(end_pc).unsqueeze(0).float().cuda())
                                # debug_and_save(start_pc, end_pc, R_cuda, t_cuda)
                            else:
                                end_offset = np.min(end_pc, axis=0)

                            eff_grasp_poses = f[side]['release'][()]

                        conditional_pc, start_offset = self.centralize_cond_pc(start_pc, self.is_obj_centric)

                        pc_tensor = torch.tensor(conditional_pc).unsqueeze(0).to(torch.float32)
                        tensor_dict[side+'_pc'] = pc_tensor

                        for pred_grasp_id in range(len(pred_grasp_poses)):
                            pred_grasp = pred_grasp_poses[pred_grasp_id].copy()

                            if self.is_obj_centric:
                                pred_grasp[:3, 3] -= start_offset
                            grasp_tensor = torch.tensor(pred_grasp).to(torch.float32).reshape(1, 4, 4)
                            
                            if self.has_eff_dict[side]:
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

                            tensor_dict[side+'_grasp'].append(grasp_tensor)

                        # assert len(joint_data_dual) > len(pred_grasp_poses)

                    for i in range(len(dual_jpose_data)):
                        left_jpose = dual_jpose_data[i][:self.dof]
                        right_jpose = dual_jpose_data[i][self.dof:]
                        joint_pose = np.vstack((left_jpose[:self.dof], right_jpose[:self.dof])).reshape(1, -1, self.dof)
                        jpose_tensor = torch.tensor(joint_pose).to(torch.float32)
                        tensor_dict['dual_jpose'].append(jpose_tensor)

                    list_entries = ['left_grasp', 'right_grasp','dual_jpose']
                    max_len_entry = max(list_entries, key=lambda x: len(tensor_dict[x]))
                    other_entires = list(set(list_entries) - set([max_len_entry]))
                    pc_entries = list(set(tensor_dict.keys()) - set(list_entries))
                    for i in range(len(tensor_dict[max_len_entry])):
                        data = {}
                        data[max_len_entry] = tensor_dict[max_len_entry][i]
                        for entry in pc_entries:
                            data[entry] = tensor_dict[entry]
                        for entry in other_entires:
                            eid = np.random.randint(0, len(tensor_dict[entry])-1)
                            data[entry] = tensor_dict[entry][eid]

                        data_list.append(data)

        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 file!')
