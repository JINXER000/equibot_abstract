from abstract_dataset import ALOHAPoseDataset, \
    solve_pairwise_registration, VecDGCNN_att_frozen,\
    rotate_around_z, rotate_vec_grasp, rotate_observation,\
    to_tensor, render_pose
import os
import numpy as np
import torch
import hydra
import pathlib
EQUIBOT_PATH = pathlib.Path(__file__).parent.parent.parent.parent.absolute()

class DualAbsDataset(ALOHAPoseDataset):
    def __init__(self, cfg, mode, transform=None, **kwargs):
        self.has_eff_dict = {'left': False, 'right': False}
        hand_sides = ['left', 'right']
        for i in range(len(hand_sides)):
            self.has_eff_dict[hand_sides[i]] = cfg.has_eff_list[i]
            
        super(DualAbsDataset, self).__init__(cfg, mode, transform, **kwargs)

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
        traj_len = cfg.pred_horizon
        traj_nums = 64
        
        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]
            
            if 'hdf5' in  file_name:
                hdf5_path = os.path.join(self.root, 'raw', file_name)
                import h5py
                with h5py.File(hdf5_path, 'r') as f:

                    socket_pc = f['socket_grasps']['obj_points'][()]
                    peg_pc = f['peg_grasps']['obj_points'][()]
                    all_socket_grasps = f['socket_grasps']['grasp_poses'][()]
                    all_peg_grasps = f['peg_grasps']['grasp_poses'][()]
                    all_socket_gripper = f['socket_grasps']['grasp_actions'][()]
                    all_peg_gripper = f['peg_grasps']['grasp_actions'][()]
                    all_joint_data = f['pred_joint_vals'][()]

                    assert len(all_socket_grasps) == len(all_socket_gripper)
                    assert len(all_peg_grasps) == len(all_peg_gripper)
                    ## downsample and centralize pc
                    socket_pc_n, socket_offset = self.centralize_cond_pc(socket_pc)
                    socket_pc_tensor = torch.tensor(socket_pc_n).unsqueeze(0).\
                        to(torch.float32).reshape(1, cfg.num_points, 3)

                    peg_pc_n, peg_offset = self.centralize_cond_pc( peg_pc)
                    peg_pc_tensor = torch.tensor(peg_pc_n).unsqueeze(0).\
                        to(torch.float32).reshape(1, cfg.num_points, 3)

                    ### random select n groups of grasp and joint pose
                    for i in range(traj_nums):
                        assert len(all_joint_data) > 1
                        # if len(all_joint_data) < traj_len:
                        #     qtraj_indices = np.linspace(0, len(all_joint_data)-1, traj_len).astype(int)
                        # else:
                        #     qtraj_indices = np.random.choice(len(all_joint_data), traj_len, replace=False)
                        #     qtraj_indices = np.sort(qtraj_indices)
                        # joint_data = all_joint_data[qtraj_indices]
                        qtraj_indice = np.random.randint(0, len(all_joint_data)-1)
                        joint_data = all_joint_data[qtraj_indice].reshape(1, -1)

                        gr_traj_indices = np.random.choice(np.arange(1, len(all_socket_grasps)-1), traj_len-2, replace=False)
                        gr_traj_indices = [0] + list(np.sort(gr_traj_indices)) + [len(all_socket_grasps)-1]
                        socket_grasp_poses = all_socket_grasps[gr_traj_indices]
                        socket_grasp_poses[:, :3, 3] -= np.expand_dims(socket_offset, axis=0)
                        socket_grasp_tensor = torch.tensor(socket_grasp_poses).to(torch.float32)


                        gl_traj_indices = np.random.choice(np.arange(1, len(all_peg_grasps)-1), traj_len-2, replace=False)
                        gl_traj_indices = [0] + list(np.sort(gl_traj_indices)) + [len(all_peg_grasps)-1]
                        peg_grasp_poses = all_peg_grasps[gl_traj_indices]
                        peg_grasp_poses[:, :3, 3] -= np.expand_dims(peg_offset, axis=0)
                        peg_grasp_tensor = torch.tensor(peg_grasp_poses).to(torch.float32)

                        socket_gripper = all_socket_gripper[gr_traj_indices].reshape(traj_len, -1)
                        socket_gripper_tensor = torch.tensor(socket_gripper).to(torch.float32)
                        peg_gripper = all_peg_gripper[gl_traj_indices].reshape( traj_len, -1)
                        peg_gripper_tensor = torch.tensor(peg_gripper).to(torch.float32)

                        left_jpose = joint_data[:, :7]                        
                        right_jpose = joint_data[:, 7:]
                        # left_jpose_tensor = torch.tensor(left_jpose).to(torch.float32).reshape(traj_len, 1, -1)
                        # right_jpose_tensor = torch.tensor(right_jpose).to(torch.float32).reshape(traj_len, 1, -1)
                        # dual_jpose_tensor = torch.cat((left_jpose_tensor, right_jpose_tensor), dim=1) # traj_len, 2, 7 
                        left_jpose_tensor = torch.tensor(left_jpose).to(torch.float32).reshape(1, 1, -1)
                        right_jpose_tensor = torch.tensor(right_jpose).to(torch.float32).reshape(1, 1, -1)
                        dual_jpose_tensor = torch.cat((left_jpose_tensor, right_jpose_tensor), dim=1) # traj_len, 2, 7 

                        data = {'left_jpose': left_jpose_tensor, 'right_jpose': right_jpose_tensor,\
                                    'left_pc': socket_pc_tensor, 'right_pc': peg_pc_tensor,\
                                    'left_grasp': socket_grasp_tensor, 'right_grasp': peg_grasp_tensor,
                                    'left_gripper': socket_gripper_tensor, 'right_gripper': peg_gripper_tensor,
                                    'dual_jpose': dual_jpose_tensor}
                        data_list.append(data)

                    # for i in range(len(joint_data)):
                    #     left_jpose = joint_data[i][:6]
                    #     right_jpose = joint_data[i][7:13]

                    #     ## downsample and centralize pc
                    #     socket_pc_n, socket_offset = self.centralize_cond_pc(socket_pc)
                    #     socket_pc_tensor = torch.tensor(socket_pc_n).unsqueeze(0).\
                    #         to(torch.float32).reshape(1, cfg.num_points, 3)

                    #     peg_pc_n, peg_offset = self.centralize_cond_pc( peg_pc)
                    #     peg_pc_tensor = torch.tensor(peg_pc_n).unsqueeze(0).\
                    #         to(torch.float32).reshape(1, cfg.num_points, 3)

                    #     ## add gripper action (claw)
                    #     left_jpose = np.concatenate((left_jpose, np.array([joint_data[i][7]])))
                    #     left_jpose_tensor = torch.tensor(left_jpose).to(torch.float32).reshape(1, 1, -1)
                    #     right_jpose = np.concatenate((right_jpose, np.array([joint_data[i][-1]])))
                    #     right_jpose_tensor = torch.tensor(right_jpose).to(torch.float32).reshape(1, 1, -1)
                    #     dual_jpose_tensor = torch.cat((left_jpose_tensor, right_jpose_tensor), dim=1) # 1, 2, 7
                        
                    #     socket_grasp_id = np.random.randint(0, len(socket_grasp_poses)-1)
                    #     socket_grasp = socket_grasp_poses[socket_grasp_id].copy()
                    #     # socket_grasp = self.centralize_grasp(socket_grasp, socket_offset)
                    #     socket_grasp[:3, 3] -= socket_offset
                    #     socket_grasp_tensor = torch.tensor(socket_grasp).to(torch.float32).reshape(1, 4, 4)
                        
                    #     peg_grasp_id = np.random.randint(0, len(peg_grasp_poses)-1)
                    #     peg_grasp = peg_grasp_poses[peg_grasp_id].copy()
                    #     # peg_grasp = self.centralize_grasp(peg_grasp, peg_offset)
                    #     peg_grasp[:3, 3] -= peg_offset
                    #     peg_grasp_tensor = torch.tensor(peg_grasp).to(torch.float32).reshape(1, 4, 4)

                    #     if "socket" in cfg.dataset_type:
                    #         data = {'jpose': left_jpose_tensor, 
                    #                 'pc': socket_pc_tensor, 
                    #                 'grasp': socket_grasp_tensor}
                    #     elif "peg" in cfg.dataset_type:
                    #         data = {'jpose': right_jpose_tensor, 
                    #                 'pc': peg_pc_tensor, 
                    #                 'grasp': peg_grasp_tensor}
                    #     else:
                    #         data = {'left_jpose': left_jpose_tensor, 'right_jpose': right_jpose_tensor,\
                    #                 'left_pc': socket_pc_tensor, 'right_pc': peg_pc_tensor,\
                    #                 'left_grasp': socket_grasp_tensor, 'right_grasp': peg_grasp_tensor,
                    #                 'dual_jpose': dual_jpose_tensor}
                    #     data_list.append(data)

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


@hydra.main(config_path=os.path.join(EQUIBOT_PATH, "equibot/policies/configs"), config_name="transfer_tape")
def main(cfg):
    cfg.data.dataset.path=os.path.join(EQUIBOT_PATH, 'data/mj_peg_hole/')
    test_dataset = DualAbsDataset(cfg.data.dataset, "test", force_process = True)
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

            for side in ['left', 'right']:
                pc_vis_data = cpu_obs[side+'_pc'][0]
                grasp_vis_data = cpu_obs[side+'_grasp'][0]
                jpose_visdata = cpu_obs['dual_jpose'][0]

                history_list = []
                tmp_pc = pc_vis_data[0].reshape(-1, 3).numpy()
                traj_len = grasp_vis_data.shape[0]
                for i in range(traj_len):
                    jpose = jpose_visdata[0].reshape(-1).numpy()
                    grasp_pose = grasp_vis_data[i,:4].reshape(1,-1,4).numpy()
                    grasp_pose_tensor = torch.tensor(grasp_pose)

                    vecrot_grasp = rotate_vec_grasp(grasp_pose_tensor, rot_z)
                    # action_slice = (grasp, jpose)
                    action_slice = (vecrot_grasp.reshape(-1, 4), jpose)
                    history_list.append(action_slice)

                render_pose(history_list, use_gui=True, \
                            directory = None, obj_points = tmp_pc)



if __name__ == '__main__':
    main()