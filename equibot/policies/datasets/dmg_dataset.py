import os
import h5py
import networkx as nx
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from equibot.policies.vision.vdgcnn_encoder import VecDGCNN_att_frozen
from equibot.policies.datasets.effpose_estimation import solve_pairwise_registration, debug_and_save
from equibot.policies.utils.misc import rotate_around_z, rotate_observation, rotate_vec_grasp, to_tensor, to_np, EQUIBOT_PATH, str_to_ascii_tensor, ascii_tensor_to_str, get_skill_names, compose_transformation, centralize_downsample, centralize_grasp, choose_ids, rotate_dataslice, get_rbt_states, get_rbt_actions, get_pc_instances, get_sg

import hydra



class RobosuiteDataset(Dataset):
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

        self.pc_shape = (cfg.num_points, 3)
        # self.has_eff_list = cfg.has_eff_list
        # self.has_eff = True in self.has_eff_list

        self.is_obj_centric = cfg.is_obj_centric
        self.is_add_bottom = cfg.is_add_bottom
        self.downsample_method = cfg.downsample_method

        self.num_eef = cfg.num_eef
        self.dof = cfg.dof
        self.dataset_type = cfg.dataset_type

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
        return os.path.join(self.root, 'processed', f'{self.dataset_type}.pt')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
    def process_select(self, cfg, **kwargs):

        if self.dataset_type == 'robosuite_separate_hdf5_traj':
            self.process_robosuite_separate_hdf5_traj(cfg, **kwargs)
        elif self.dataset_type == 'robosuite_separate_hdf5_grasp':
            self.process_robosuite_separate_hdf5_grasp(cfg, **kwargs)
        elif self.dataset_type == 'robosuite_integrate_hdf5_traj':
            self.process_robosuite_integrate_hdf5_traj(cfg, **kwargs)
        elif self.dataset_type == 'robosuite_different_skills_traj':
            self.process_robosuite_different_skills_traj(cfg, **kwargs)
        else:
            raise NotImplementedError(f'Dataset type {self.dataset_type} not implemented!')
        
    def process_robosuite_separate_hdf5_grasp(self, cfg, **kwargs):
        print('Processing hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names
        grasp_nums = cfg.pred_horizon
        repeat_nums = 16
        interested_skills = cfg.uniskills

        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]
            if 'hdf5' not in  file_name:
                continue
        
            hdf5_path = os.path.join(self.root, 'raw', file_name)
            with h5py.File(hdf5_path, 'r') as f:
                ## read sg
                sg_info = f['sg_info']
                sg_params_json = f['sg_params'][()]
                sg_params = json.loads(sg_params_json.decode('utf-8'))
                robot_names = sg_params['robots']   

                demo_id = file_name.split('_')[-2]
                obs_grp = f[f'data/demo_{demo_id}/obs']
                rbt_states = get_rbt_states(obs_grp, robot_names)

                # ## NOTE: robot1 --> left, robot0 --> right
                # left_gripper_actions = f[f'data/demo_{demo_id}/action_dict/left_gripper'][()]
                # right_gripper_actions = f[f'data/demo_{demo_id}/action_dict/right_gripper'][()]
                # gripper_actions = {'robot1': left_gripper_actions, 'robot0': right_gripper_actions}

                obj_pcds =  {}
                obj_conditioned_skills = {}
                for obj_pc_key in f['data/obj_pcd'].keys():
                    raw_vlen = f[f'data/obj_pcd/{obj_pc_key}'][()]
                    pc_list = [raw_vlen[i].reshape(-1, 3) for i in range(len(raw_vlen))]
                    obj_name = obj_pc_key.split('_points')[0]
                    obj_pcds[obj_name] = pc_list

                    ## associate each skill with the corresponding object
                    # related_skills = [skill_name for skill_name, skill_info in sg_info.items() if obj_name in skill_info['related_objs']]
                    related_skills = []
                    for skill_name, skill_info in sg_info.items():
                        related_objs = [rel_obj.decode('utf-8') for rel_obj in skill_info['related_objs']]
                        related_skills += [skill_name for rel_obj in related_objs if obj_name == rel_obj]
                    obj_conditioned_skills[obj_name] = related_skills

                for _ in range(repeat_nums):
                    data_slice = {}

                    for obj_name, obj_pc_list in obj_pcds.items(): ## now we only reuse obj encoder. pc is not concatenated. 
                        for skill_name in obj_conditioned_skills[obj_name]:
                            skill_info = sg_info[skill_name]
                            pre_sg = get_sg(skill_info, 'pre_sg')
                            cur_sg = get_sg(skill_info, 'cur_sg')
                            eff_sg = get_sg(skill_info, 'eff_sg')
                            
                            if 'bimanual' in skill_name:
                                pre_idx_list = pre_sg.graph['idx_list']

                                pre_dual_jpose_all = np.concatenate([rbt_states['robot0_joint_pos'][pre_idx_list], \
                                    rbt_states['robot1_joint_pos'][pre_idx_list]], axis=1)
                                qtraj_indice = np.random.randint(0, len(pre_dual_jpose_all)-1)
                                data_slice[f'{skill_name}:jpose'] = pre_dual_jpose_all[qtraj_indice].astype(np.float32)

                                if 'eff_sg' in skill_info:
                                    eff_idx_list = eff_sg.graph['idx_list']
                                    eff_dual_jpose_all = np.concatenate([rbt_states['robot0_joint_pos'][eff_idx_list], \
                                        rbt_states['robot1_joint_pos'][eff_idx_list]], axis=1)
                                    qtraj_indice = np.random.randint(0, len(eff_dual_jpose_all)-1)
                                    eff_dual_jpose= eff_dual_jpose_all[qtraj_indice].astype(np.float32)
                                    data_slice[f'{skill_name}:jpose'] = np.concatenate([data_slice[f'{skill_name}:jpose'], eff_dual_jpose], axis=0)

                            else:
                                if 'grasp' in skill_name:
                                    if 'grasp' not in interested_skills:
                                        continue
                                    essential_ids = skill_info['essential_ids'][()]
                                    skill_key = 'grasp'
                                    # obj_name = skill_name.split('_', 1)[1]
                                    obj_pc = obj_pc_list[pre_sg.graph['idx_list'][0]]
                                elif 'release' in skill_name:
                                    if 'release' not in interested_skills:
                                        continue
                                    essential_ids = skill_info['essential_ids'][()]
                                    skill_key = 'release'
                                    # obj_name = skill_name.split('_', 1)[1]
                                    obj_pc = obj_pc_list[eff_sg.graph['idx_list'][-1]]
                                else:
                                    continue
                                    # raise NotImplementedError(f'Skill name {skill_name} not implemented!')
                                
                                
                                obj_pc_n, obj_offset = centralize_downsample(obj_pc, self.pc_shape, obj_centric = self.is_obj_centric, add_bottom = self.is_add_bottom, method = self.downsample_method, debug_visualize=True)
                                obj_pc_tensor = torch.tensor(obj_pc_n).unsqueeze(0).to(torch.float32).reshape(1, cfg.num_points, 3)
                                
                                rbt_name = skill_info['related_rbts'][0].decode('utf-8')

                                # idx_list = skill_info['extended_ids'][()]
                                ## delay from action to state
                                delay = 7
                                delayed_essential_ids = [(eid + delay) for eid in essential_ids]
                                choiced_ids = np.random.choice(delayed_essential_ids, size=grasp_nums, replace=True).astype(np.int32) # unsorted
                                eef_pos_list = rbt_states[f'{rbt_name}_eef_pos'][choiced_ids]
                                eef_quat_list = rbt_states[f'{rbt_name}_eef_quat'][choiced_ids]
                                eef_pos_list = list(map(compose_transformation, eef_pos_list, eef_quat_list))
                                normalized_eef_pos_list = list(map(centralize_grasp, eef_pos_list, [obj_offset]*grasp_nums))
                                normalized_eef_pos_tensor = torch.tensor(np.array(normalized_eef_pos_list)).to(torch.float32).reshape(grasp_nums, 4, 4) 

                                # gripper_list = gripper_actions[rbt_name][choiced_ids]
                                # # open_num = len(gripper_list[gripper_list < 0])
                                # # print(f"open num is: {open_num}, skill name: {skill_name}, obj name: {obj_name}")

                                gripper_list = rbt_states[f'{rbt_name}_gripper_qpos'][choiced_ids,0]
                                
                                data_slice[f'{skill_name}:pc'] = obj_pc_tensor
                                data_slice[f'{skill_name}:eefpos'] = normalized_eef_pos_tensor
                                data_slice[f'{skill_name}:gripper'] = torch.tensor(gripper_list).to(torch.float32).reshape(grasp_nums, 1, 1)
                                data_slice[f'{skill_name}:obj_name'] = str_to_ascii_tensor(obj_name)

                    if cfg.rot_aug:
                        data_slice = rotate_dataslice(data_slice)
                    data_list.append(data_slice)
        
        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 file!')


    def process_robosuite_separate_hdf5_traj(self, cfg, **kwargs):

        print('Processing hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names
        traj_len = cfg.pred_horizon
        traj_nums = 64
        interested_skills = cfg.uniskills

        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]
            if 'hdf5' not in  file_name:
                continue
        
            hdf5_path = os.path.join(self.root, 'raw', file_name)
            with h5py.File(hdf5_path, 'r') as f:
                ## read sg
                sg_info = f['sg_info']
                sg_params_json = f['sg_params'][()]
                sg_params = json.loads(sg_params_json.decode('utf-8'))
                robot_names = sg_params['robots']   

                demo_id = file_name.split('_')[-2]
                obs_grp = f[f'data/demo_{demo_id}/obs']
                rbt_states = get_rbt_states(obs_grp, robot_names)

                ## NOTE: robot1 --> left, robot0 --> right
                left_gripper_actions = f[f'data/demo_{demo_id}/action_dict/left_gripper'][()]
                right_gripper_actions = f[f'data/demo_{demo_id}/action_dict/right_gripper'][()]
                gripper_actions = {'robot1': left_gripper_actions, 'robot0': right_gripper_actions}

                obj_pcds =  {}
                obj_conditioned_skills = {}
                for obj_pc_key in f['data/obj_pcd'].keys():
                    raw_vlen = f[f'data/obj_pcd/{obj_pc_key}'][()]
                    pc_list = [raw_vlen[i].reshape(-1, 3) for i in range(len(raw_vlen))]
                    obj_name = obj_pc_key.split('_points')[0]
                    obj_pcds[obj_name] = pc_list

                    ## associate each skill with the corresponding object
                    # related_skills = [skill_name for skill_name, skill_info in sg_info.items() if obj_name in skill_info['related_objs']]
                    related_skills = []
                    for skill_name, skill_info in sg_info.items():
                        related_objs = [rel_obj.decode('utf-8') for rel_obj in skill_info['related_objs']]
                        related_skills += [skill_name for rel_obj in related_objs if obj_name == rel_obj]
                    obj_conditioned_skills[obj_name] = related_skills

                for _ in range(traj_nums):
                    data_slice = {}

                    for obj_name, obj_pc_list in obj_pcds.items(): ## now we only reuse obj encoder. pc is not concatenated. 
                        for skill_name in obj_conditioned_skills[obj_name]:
                            skill_info = sg_info[skill_name]
                            pre_sg = get_sg(skill_info, 'pre_sg')
                            cur_sg = get_sg(skill_info, 'cur_sg')
                            eff_sg = get_sg(skill_info, 'eff_sg')
                            
                            if 'bimanual' in skill_name:
                                pre_idx_list = pre_sg.graph['idx_list']

                                pre_dual_jpose_all = np.concatenate([rbt_states['robot0_joint_pos'][pre_idx_list], \
                                    rbt_states['robot1_joint_pos'][pre_idx_list]], axis=1)
                                qtraj_indice = np.random.randint(0, len(pre_dual_jpose_all)-1) if len(pre_dual_jpose_all) > 1 else 0
                                data_slice[f'{skill_name}:jpose'] = pre_dual_jpose_all[qtraj_indice].astype(np.float32)

                                if 'eff_sg' in skill_info:
                                    eff_idx_list = eff_sg.graph['idx_list']
                                    eff_dual_jpose_all = np.concatenate([rbt_states['robot0_joint_pos'][eff_idx_list], \
                                        rbt_states['robot1_joint_pos'][eff_idx_list]], axis=1)
                                    qtraj_indice = np.random.randint(0, len(eff_dual_jpose_all)-1)
                                    eff_dual_jpose= eff_dual_jpose_all[qtraj_indice].astype(np.float32)
                                    data_slice[f'{skill_name}:jpose'] = np.concatenate([data_slice[f'{skill_name}:jpose'], eff_dual_jpose], axis=0)

                            else:
                                if 'grasp' in skill_name:
                                    if 'grasp' not in interested_skills:
                                        continue
                                    essential_ids = skill_info['essential_ids'][()]
                                    skill_key = 'grasp'
                                    # obj_name = skill_name.split('_', 1)[1]
                                    obj_pc = obj_pc_list[pre_sg.graph['idx_list'][0]]
                                elif 'contact' in skill_name:
                                    if 'contact' not in interested_skills:
                                        continue
                                    essential_ids = None
                                    skill_key = 'contact'
                                    # obj_name = skill_name.split('_contact_', 1)[1]
                                    obj_pc = obj_pc_list[cur_sg.graph['idx_list'][0]]
                                elif 'release' in skill_name:
                                    if 'release' not in interested_skills:
                                        continue
                                    essential_ids = skill_info['essential_ids'][()]
                                    skill_key = 'release'
                                    # obj_name = skill_name.split('_', 1)[1]
                                    obj_pc = obj_pc_list[eff_sg.graph['idx_list'][-1]]
                                else:
                                    continue
                                    # raise NotImplementedError(f'Skill name {skill_name} not implemented!')
                                
                                
                                obj_pc_n, obj_offset = centralize_downsample(obj_pc, self.pc_shape, obj_centric = self.is_obj_centric, add_bottom = self.is_add_bottom, method = self.downsample_method, debug_visualize=True)
                                obj_pc_tensor = torch.tensor(obj_pc_n).unsqueeze(0).to(torch.float32).reshape(1, cfg.num_points, 3)
                                
                                rbt_name = skill_info['related_rbts'][0].decode('utf-8')

                                # idx_list = cur_sg.graph['idx_list']
                                idx_list = skill_info['extended_ids'][()]

                                # delay = 7
                                # delayed_essential_ids = [(eid + delay) for eid in essential_ids]
                                choiced_ids = choose_ids(traj_len, idx_list, essential_ids, skill_key)
                                eef_pos_list = rbt_states[f'{rbt_name}_eef_pos'][choiced_ids]
                                eef_quat_list = rbt_states[f'{rbt_name}_eef_quat'][choiced_ids]
                                eef_pos_list = list(map(compose_transformation, eef_pos_list, eef_quat_list))
                                normalized_eef_pos_list = list(map(centralize_grasp, eef_pos_list, [obj_offset]*traj_len))
                                normalized_eef_pos_tensor = torch.tensor(normalized_eef_pos_list).to(torch.float32).reshape(traj_len, 4, 4) 

                                gripper_list = gripper_actions[rbt_name][choiced_ids]
                                # open_num = len(gripper_list[gripper_list < 0])
                                # print(f"open num is: {open_num}, skill name: {skill_name}, obj name: {obj_name}")
                                
                                data_slice[f'{skill_name}:pc'] = obj_pc_tensor
                                data_slice[f'{skill_name}:eefpos'] = normalized_eef_pos_tensor
                                data_slice[f'{skill_name}:gripper'] = torch.tensor(gripper_list).to(torch.float32).reshape(traj_len, 1, 1)
                                data_slice[f'{skill_name}:obj_name'] = str_to_ascii_tensor(obj_name)

                    if cfg.rot_aug:
                        data_slice = rotate_dataslice(data_slice)
                    data_list.append(data_slice)
        
        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 file!')

    def process_robosuite_integrate_hdf5_traj(self, cfg, **kwargs):

        print('Processing hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names
        traj_len = cfg.pred_horizon
        traj_nums = 32
        primitive_kws = cfg.uniskills
        interested_objs = cfg.conditioned_objects
        skill_names = cfg.skill_names
        skill_condition_objs = {skill_names[i]: interested_objs[i] for i in range(len(skill_names))}
        self.involved_skill_names = set()

        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]
            if 'hdf5' not in  file_name:
                continue
        
            hdf5_path = os.path.join(self.root, 'raw', file_name)
            with h5py.File(hdf5_path, 'r') as f:
                ## read sg
                sg_params_json = f['sg_params'][()]
                sg_params = json.loads(sg_params_json.decode('utf-8'))
                robot_names = sg_params['robots']  


                demos = [ent for ent in list(f['data'].keys()) if ent.startswith('demo_')]
                inds = np.argsort([int(elem[5:]) for elem in demos])
                demos = [demos[i] for i in inds]

                n_use = cfg.n_use if 'n_use' in cfg else len(demos)
                demos = demos[:n_use]

                for demo_id in range(len(demos)):
                    sg_info = f[f'data/demo_{demo_id}/sg_info']
                
                    obs_grp = f[f'data/demo_{demo_id}/obs']
                    rbt_states = get_rbt_states(obs_grp, robot_names)
                    obj_pcds = get_pc_instances(obs_grp, interested_objs)
                    action_arr = f[f'data/demo_{demo_id}/actions'][()]
                    rbt_action = get_rbt_actions(action_arr, robot_names)

                    if len(sg_info) < 2:
                        ## skip if place skills for libero
                        print(f"Skip demo {demo_id} due to insufficient skills.")
                        continue
                    for _ in range(traj_nums):

                        data_slice = {}
                        for skill_name in skill_names:
                            skill_info = sg_info[skill_name]
                            self.involved_skill_names.add(skill_name)

                            pre_sg = get_sg(skill_info, 'pre_sg')
                            cur_sg = get_sg(skill_info, 'cur_sg')
                            eff_sg = get_sg(skill_info, 'eff_sg')

                            if 'bimanual' in skill_name:
                                pre_idx_list = pre_sg.graph['idx_list']
                                pre_left_eef_pos = rbt_states['robot0_eef_pos'][pre_idx_list]
                                pre_right_eef_pos = rbt_states['robot1_eef_pos'][pre_idx_list]
                                pre_eef_dist = np.linalg.norm(pre_left_eef_pos - pre_right_eef_pos, axis=1)
                                ## gfilter idx by eef dist
                                max_eef_dist = 0.3  
                                min_eef_dist = 0.25
                                distclose_ids = list(set(np.where(pre_eef_dist < max_eef_dist)[0]).intersection(np.where(pre_eef_dist >min_eef_dist)[0]))
                                qtraj_indice = pre_idx_list[np.random.choice(distclose_ids)]
                                selected_jpose = np.concatenate([rbt_states['robot0_joint_pos'][qtraj_indice], rbt_states['robot1_joint_pos'][qtraj_indice]], axis=0).astype(np.float32)
                                data_slice[f'{skill_name}:jpose'] = selected_jpose
                                # pre_dual_jpose_all = np.concatenate([rbt_states['robot0_joint_pos'][pre_idx_list], \
                                #     rbt_states['robot1_joint_pos'][pre_idx_list]], axis=1)
                                # qtraj_indice = np.random.randint(0, len(pre_dual_jpose_all)-1) if len(pre_dual_jpose_all) > 1 else 0
                                # data_slice[f'{skill_name}:jpose'] = pre_dual_jpose_all[qtraj_indice].astype(np.float32)

                                # if 'eff_sg' in skill_info:
                                #     eff_idx_list = eff_sg.graph['idx_list']
                                #     eff_dual_jpose_all = np.concatenate([rbt_states['robot0_joint_pos'][eff_idx_list], \
                                #         rbt_states['robot1_joint_pos'][eff_idx_list]], axis=1)
                                #     qtraj_indice = np.random.randint(0, len(eff_dual_jpose_all)-1)
                                #     eff_dual_jpose= eff_dual_jpose_all[qtraj_indice].astype(np.float32)
                                #     data_slice[f'{skill_name}:jpose'] = np.concatenate([data_slice[f'{skill_name}:jpose'], eff_dual_jpose], axis=0)

                            else:
                                for skill_key in primitive_kws:
                                    if skill_key in skill_name:
                                        break
                                else:
                                    ## if no interested skill found, skip this skill
                                    continue

                                # obj_name = skill_info['related_objs'][0].decode('utf-8')
                                obj_name = skill_condition_objs[skill_name]
                                obj_pc_list = obj_pcds[obj_name] 

                                essential_ids = skill_info['essential_ids'][()]
                                obj_pc = obj_pc_list[pre_sg.graph['idx_list'][0]][:, :3]
                               
                                obj_pc_n, obj_offset = centralize_downsample(obj_pc, self.pc_shape, obj_centric = self.is_obj_centric, add_bottom = self.is_add_bottom, method = self.downsample_method, debug_visualize=True)
                                obj_pc_tensor = torch.tensor(obj_pc_n).unsqueeze(0).to(torch.float32).reshape(1, cfg.num_points, 3)
                                
                                rbt_name = skill_info['related_rbts'][0].decode('utf-8')

                                idx_list = skill_info['extended_ids'][()]

                                choiced_ids = choose_ids(traj_len, idx_list, essential_ids, skill_key)
                                eef_pos_list = rbt_states[f'{rbt_name}_eef_pos'][choiced_ids]
                                eef_quat_list = rbt_states[f'{rbt_name}_eef_quat'][choiced_ids]
                                eef_pos_list = list(map(compose_transformation, eef_pos_list, eef_quat_list))
                                normalized_eef_pos_list = list(map(centralize_grasp, eef_pos_list, [obj_offset]*traj_len))
                                normalized_eef_pos_tensor = torch.tensor(normalized_eef_pos_list).to(torch.float32).reshape(traj_len, 4, 4) 

                                gripper_list = rbt_action[rbt_name][choiced_ids]
                                
                                data_slice[f'{skill_name}:pc'] = obj_pc_tensor
                                data_slice[f'{skill_name}:eefpos'] = normalized_eef_pos_tensor
                                data_slice[f'{skill_name}:gripper'] = torch.tensor(gripper_list).to(torch.float32).reshape(traj_len, 1, 1)
                                # data_slice[f'{skill_name}:obj_name'] = str_to_ascii_tensor(obj_name)

                        # ## TODO: make one slice only for one skill
                        # expected_slice_len = len(primitive_kws)*  3
                        # if len(data_slice) != expected_slice_len:
                        #     continue
                        if cfg.rot_aug:
                            data_slice = rotate_dataslice(data_slice)
                        data_list.append(data_slice)
        
        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 files!')

        cfg.skill_names = list(self.involved_skill_names)
        print(f'Involved skill names: {cfg.skill_names}')




@hydra.main(config_path=os.path.join(EQUIBOT_PATH, "equibot/policies/configs"), config_name="libero_spatial")
def main(cfg):
    import sys
    sys.path.append('/home/user/yzchen_ws/TAMP-ubuntu22/pddlstream_aloha')
    # sys.path.append('/mnt/TAMP/interbotix_ws/src/pddlstream_aloha')
    # sys.path.append('/home/xuhang/interbotix_ws/src/pddlstream_aloha')
    from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose

    test_dataset = RobosuiteDataset(cfg.data.dataset, "test", force_process = True)
    num_workers = 0
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    
    for batch_id, batch in enumerate(test_loader):
        # rot_list = [0, np.pi/2, np.pi, np.pi/2*3]
        rot_list = [0]
        for rot_z in rot_list:
            np_obs= rotate_observation(batch, rot_z)
            cpu_obs = to_tensor(np_obs)

            skill_names = get_skill_names(cpu_obs)

            for skill_name in skill_names:
                if 'bimanual' in skill_name:
                    ## vis non-prehension skills
                    history_list = []
                    jpose_data = cpu_obs[skill_name+':jpose']
                    for i in range(jpose_data.shape[0]):
                        jpose = jpose_data[i].reshape(-1)
                        action_slice = (None, jpose)
                        history_list.append(action_slice)

                    render_pose(history_list, use_gui=True, \
                                        directory = None, 
                                        robot_name = 'panda_dual')

                else:
                    ## vis prehensile skills
                    pc_vis_data = cpu_obs[skill_name+':pc'][0]
                    grasp_vis_data = cpu_obs[skill_name+':eefpos'][0]

                    history_list = []
                    tmp_pc = pc_vis_data[0].reshape(-1, 3).numpy()
                    traj_len = grasp_vis_data.shape[0]
                    for i in range(traj_len):
                        grasp_pose = grasp_vis_data[i,:4].reshape(1,-1,4).numpy()
                        grasp_pose_tensor = torch.tensor(grasp_pose)

                        vecrot_grasp = rotate_vec_grasp(grasp_pose_tensor, rot_z)
                        action_slice = (vecrot_grasp.reshape(-1, 4), None)
                        history_list.append(action_slice)

                    print(f"Rendering skill: {skill_name}")
                    render_pose(history_list, use_gui=True, \
                                directory = None, obj_points = tmp_pc,
                                robot_name = 'panda')



if __name__ == '__main__':
    main()

                        
                        
