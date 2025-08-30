import os
import h5py
import hydra
import networkx as nx
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from equibot.policies.vision.vdgcnn_encoder import VecDGCNN_att_frozen
from equibot.policies.datasets.effpose_estimation import solve_pairwise_registration, debug_and_save
from equibot.policies.utils.misc import rotate_around_z, rotate_observation, rotate_vec_grasp, to_tensor, to_np, EQUIBOT_PATH, get_skill_names, compose_transformation, centralize_downsample, centralize_grasp, choose_ids, choose_ids_rdp, rotate_dataslice, get_rbt_states, get_rbt_actions, get_pc_instances, get_sg, convert_trans_to_vec, convert_trans_to_4pts, str_to_ascii_tensor, combined_pc_instances_and_offset

from equibot.policies.utils.lan_utils import get_embs_without_saving, save_embs

from equibot.policies.utils.normalize_utils import to_torch_stats, get_torch_range_symmetric_normalizer_from_stat

from equibot.policies.utils.normalizer import LinearNormalizer


def collate_fn(batch):
    """
    Custom collate function to handle variable-length skill name tensors.
    Pads skill_name tensors to the same length for batching.
    """
    if "skill_name" not in batch[0]:
        return torch.utils.data.dataloader.default_collate(batch)
    
    # Ensure both name tensors are 1-D integer tensors with a consistent dtype
    for item in batch:
        if not torch.is_tensor(item['skill_name']):
            item['skill_name'] = torch.tensor(item['skill_name'], dtype=torch.long)
        else:
            item['skill_name'] = item['skill_name'].to(dtype=torch.long)
        if not torch.is_tensor(item['task_name']):
            item['task_name'] = torch.tensor(item['task_name'], dtype=torch.long)
        else:
            item['task_name'] = item['task_name'].to(dtype=torch.long)
    
    # Find the maximum length of skill_name/task_name tensors in the batch
    max_skill_name_len = max(len(item['skill_name']) for item in batch)
    max_task_name_len = max(len(item['task_name']) for item in batch)
    
    # Pad all skill_name/task_name tensors to the same length
    for item in batch:
        skill_name_len = len(item['skill_name'])
        if skill_name_len < max_skill_name_len:
            padding = torch.zeros(max_skill_name_len - skill_name_len, dtype=torch.long)
            item['skill_name'] = torch.cat([item['skill_name'], padding])

        task_name_len = len(item['task_name'])
        if task_name_len < max_task_name_len:
            padding = torch.zeros(max_task_name_len - task_name_len, dtype=torch.long)
            item['task_name'] = torch.cat([item['task_name'], padding])
    
    # Use default collate for the rest
    return torch.utils.data.dataloader.default_collate(batch)

def get_libero_task_emb(task_suite_name, cache_dir):
    from libero.libero import benchmark
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    descriptions = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        descriptions.append(task.name)

    task_emb_dict = get_embs_without_saving(descriptions, cache_dir=cache_dir)
    # task_emb_dict = {descriptions[i]: task_embs_list[i] for i in range(len(descriptions))}
    return task_emb_dict

def find_correct_task_name(task_name_list, hdf5_name):
    for task_name in task_name_list:
        if task_name in hdf5_name:
            return task_name
    return None

class PerSkillDataset(Dataset):
    def __init__(self, cfg, mode, transform=None, pre_transform=None, pre_filter=None,  **kwargs):
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

        self.eef_representation = cfg.eef_representation
        self.original_gripper_pcd = np.array(cfg.original_gripper_pcd)

        self.statistics = {}
        # self.skill_names = None
        # self.task_names = None

        # self.process_select(cfg,**kwargs)
        if mode == 'train':
            # Process the data
            print('Processing dataset...')
            self.process_select(cfg,**kwargs)
            self.skill_names = list(self.statistics['skill_embs_all_tasks'].keys())
            self.task_names = list(self.statistics['task_emb_dict'].keys())

        else:
            self.data = None
            self.normalizer = None


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

        if self.dataset_type == 'per_skill_libero_traj':
            ## libero
            self.data = self.process_per_skill_libero_traj(cfg, **kwargs)
            self.normalizer = self.get_normalizer_and_statistics(self.data)
        elif self.dataset_type == 'per_skill_dmg_traj':
            ## dexmimicgen
            self.data = self.process_per_skill_dmg_traj(cfg, **kwargs)
            self.normalizer = self.get_normalizer_and_statistics(self.data)
        elif self.dataset_type == 'per_skill_biop_jpose':
            self.data = self.process_per_biop(cfg, **kwargs)
            self.normalizer = self.get_normalizer_and_statistics(self.data, mode = 'bimanual')
        else:
            raise NotImplementedError(f'Dataset type {self.dataset_type} not implemented!')
        

    def process_per_skill_libero_traj(self, cfg, **kwargs):

        print('Processing hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names
        traj_len = cfg.pred_horizon
        traj_nums = 32
        primitive_kws = cfg.uniskills
        task_suite_name = cfg.task_suite_name
        cache_dir = os.path.join(EQUIBOT_PATH, cfg.embedding_cache_dir)
        task_emb_dict = get_libero_task_emb(task_suite_name, cache_dir)
        # interested_objs = cfg.conditioned_objects
        # skill_names = cfg.skill_names
        # skill_condition_objs = {skill_names[i]: interested_objs[i] for i in range(len(skill_names))}
        self.involved_skill_names = set()
        skill_embs_all_tasks = {}
        matched_action_sgs = {}

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

                ## get task name and emb for libero
                # task_name = sg_params['task_name']
                task_name = find_correct_task_name(task_emb_dict.keys(), file_name)
                # task_desc = task_name.replace('_', ' ')
                # task_emb = task_emb_dict[task_name]

                demos = [ent for ent in list(f['data'].keys()) if ent.startswith('demo_')]   
                inds = np.argsort([int(elem[5:]) for elem in demos])
                demos = [demos[i] for i in inds]

                n_use = cfg.n_use if 'n_use' in cfg else len(demos)
                demos = demos[:n_use]

                ## record the skillwise_sgs
                matched_action_sgs[task_name] = f[f'data/demo_0/matched_actions_json'][()]

                ## interested objs and skills for each task
                interested_objs = set()
                interested_skills = set()
                ## get all skill names
                for demo_id in range(len(demos)):
                    sg_info = f[f'data/demo_{demo_id}/sg_info']

                    for skill_name in sg_info.keys():
                        for skill_key in primitive_kws:
                            if skill_key in skill_name:
                                interested_skills.add(skill_name)
                                skill_info = sg_info[skill_name]
                                interested_objs.add(skill_info['related_objs'][0].decode('utf-8'))
                                break
                        else:
                            ## if no interested skill found, skip this skill
                            continue

                skill_name_to_emb = get_embs_without_saving(list(interested_skills), cache_dir=cache_dir)
                skill_embs_all_tasks.update(skill_name_to_emb)
                self.involved_skill_names = self.involved_skill_names.union(interested_skills)

                for demo_id in range(len(demos)):
                    sg_info = f[f'data/demo_{demo_id}/sg_info']
                
                    obs_grp = f[f'data/demo_{demo_id}/obs']
                    rbt_states = get_rbt_states(obs_grp, robot_names)
                    obj_pcds = get_pc_instances(obs_grp, interested_objs)
                    action_arr = f[f'data/demo_{demo_id}/actions'][()]
                    rbt_action = get_rbt_actions(action_arr, robot_names)


                    for _ in range(traj_nums):
                
                        # Create separate data slices for each skill name
                        for skill_name, skill_info in sg_info.items():

                            ## filter out the bad demo
                            essential_ids = skill_info['essential_ids'][()]
                            if len(essential_ids) == 0:
                                print(f'No essential ids found for {skill_name} in {task_name}, demo {demo_id}')
                                continue

                            ## only use interested unimanual skills
                            if skill_name not in interested_skills:
                                continue

                            data_slice  = self.get_dataslice_unimanual(skill_info, skill_name, skill_key, cfg, traj_len, obj_pcds, rbt_states, rbt_action, task_name)

                            data_list.append(data_slice)
        
        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 files!')

        # cfg.skill_names = list(self.involved_skill_names)
        # print(f'Involved skill names: {cfg.skill_names}')
        ## obtain skill name embedding
        cache_name = f'{cfg.dataset_type}_skill_name_to_emb.npy'
        save_embs(skill_embs_all_tasks, cache_dir=cache_dir, cache_name=cache_name)

        self.statistics['task_emb_dict'] = task_emb_dict
        self.statistics['skill_embs_all_tasks'] = skill_embs_all_tasks
        self.statistics['matched_action_sgs'] = matched_action_sgs
        return data_list

    def process_per_biop(self, cfg, **kwargs):
        print('Processing hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names
        traj_len = cfg.pred_horizon
        traj_nums = 32
        primitive_kws = cfg.uniskills
        # task_suite_name = cfg.task_suite_name
        cache_dir = os.path.join(EQUIBOT_PATH, cfg.embedding_cache_dir)
        # task_emb_dict = get_libero_task_emb(task_suite_name, cache_dir)
        self.involved_skill_names = set()
        skill_embs_all_tasks = {}
        matched_action_sgs = {}

        involved_tasks = set()

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

                ## get task name and emb for libero
                task_name = sg_params['task_name']
                involved_tasks.add(task_name)
                # task_name = find_correct_task_name(task_emb_dict.keys(), file_name)


                demos = [ent for ent in list(f['data'].keys()) if ent.startswith('demo_')]   
                inds = np.argsort([int(elem[5:]) for elem in demos])
                demos = [demos[i] for i in inds]

                n_use = cfg.n_use if 'n_use' in cfg else len(demos)
                demos = demos[:n_use]

                ## record the skillwise_sgs
                matched_action_sgs[task_name] = f[f'data/demo_0/matched_actions_json'][()]

                ## interested objs and skills for each task
                interested_objs = set()
                interested_skills = set()
                for demo_id in range(len(demos)):
                    sg_info = f[f'data/demo_{demo_id}/sg_info']
                    for skill_name in sg_info.keys():
                        if 'bi' in skill_name:
                            interested_skills.add(skill_name)
                            skill_info = sg_info[skill_name]
                            interested_objs.add(skill_info['related_objs'][0].decode('utf-8'))
                            break
                    else:
                        ## if no interested skill found, skip this skill
                        continue

                skill_name_to_emb = get_embs_without_saving(list(interested_skills), cache_dir=cache_dir)
                skill_embs_all_tasks.update(skill_name_to_emb)
                self.involved_skill_names = self.involved_skill_names.union(interested_skills)

                for demo_id in range(len(demos)):
                    sg_info = f[f'data/demo_{demo_id}/sg_info']
                
                    obs_grp = f[f'data/demo_{demo_id}/obs']
                    rbt_states = get_rbt_states(obs_grp, robot_names)
                    obj_pcds = get_pc_instances(obs_grp, interested_objs)
                    action_arr = f[f'data/demo_{demo_id}/actions'][()]
                    rbt_action = get_rbt_actions(action_arr, robot_names)

                    for _ in range(traj_nums):
                        # Create separate data slices for each skill name
                        for skill_name, skill_info in sg_info.items():

                            ## only use bimanual skills
                            if 'bimanual' in skill_name:
                                data_slice = self.get_dataslice_bimanual_jpose(skill_info, skill_name, rbt_states, task_name)
                            else:
                                continue
                                   
                            data_list.append(data_slice)
        


        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 files!')

        cache_name = f'{cfg.dataset_type}_skill_name_to_emb.npy'
        save_embs(skill_embs_all_tasks, cache_dir=cache_dir, cache_name=cache_name)

        task_emb_dict = get_embs_without_saving(list(involved_tasks), cache_dir=cache_dir)
        self.statistics['task_emb_dict'] = task_emb_dict
        self.statistics['skill_embs_all_tasks'] = skill_embs_all_tasks
        self.statistics['matched_action_sgs'] = matched_action_sgs

        return data_list


    def get_dataslice_bimanual_jpose(self, skill_info, skill_name,  rbt_states,  task_name):
        pre_sg = get_sg(skill_info, 'pre_sg')
        data_slice_bi = {}
        pre_idx_list = pre_sg.graph['idx_list']
        pre_left_eef_pos = rbt_states['robot0_eef_pos'][pre_idx_list]
        pre_right_eef_pos = rbt_states['robot1_eef_pos'][pre_idx_list]
        pre_eef_dist = np.linalg.norm(pre_left_eef_pos - pre_right_eef_pos, axis=1)
        ## gfilter idx by eef dist
        max_eef_dist = 0.29 # 0.5 #0.29
        min_eef_dist = 0.25 # 0.3 #0.25
        distclose_ids = list(set(np.where(pre_eef_dist < max_eef_dist)[0]).intersection(np.where(pre_eef_dist >min_eef_dist)[0]))
        if len(distclose_ids) == 0:
            print(f'No valid bimanual jpose found for {skill_name} in {task_name}')
            return None

        qtraj_indice = pre_idx_list[np.random.choice(distclose_ids)]
        selected_jpose = np.concatenate([rbt_states['robot0_joint_pos'][qtraj_indice], rbt_states['robot1_joint_pos'][qtraj_indice]], axis=0).astype(np.float32)
        data_slice_bi['jpose'] = selected_jpose
        data_slice_bi['skill_name'] = str_to_ascii_tensor(skill_name)
        data_slice_bi['task_name'] = str_to_ascii_tensor(task_name)

        return data_slice_bi

    def process_per_skill_dmg_traj(self, cfg, **kwargs):

        print('Processing hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names
        traj_len = cfg.pred_horizon
        traj_nums = 32
        primitive_kws = cfg.uniskills
        # task_suite_name = cfg.task_suite_name
        cache_dir = os.path.join(EQUIBOT_PATH, cfg.embedding_cache_dir)
        # task_emb_dict = get_libero_task_emb(task_suite_name, cache_dir)
        self.involved_skill_names = set()
        skill_embs_all_tasks = {}
        matched_action_sgs = {}
        involved_tasks = set()

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

                ## get task name and emb for libero
                task_name = sg_params['task_name']
                involved_tasks.add(task_name)
                # task_name = find_correct_task_name(task_emb_dict.keys(), file_name)

                ## record the skillwise_sgs
                matched_action_sgs[task_name] = f[f'data/demo_0/matched_actions_json'][()]

                demos = [ent for ent in list(f['data'].keys()) if ent.startswith('demo_')]   
                inds = np.argsort([int(elem[5:]) for elem in demos])
                demos = [demos[i] for i in inds]

                n_use = cfg.n_use if 'n_use' in cfg else len(demos)
                demos = demos[:n_use]

                ## interested objs and skills for each task
                interested_objs = set()
                interested_skills = set()
                ## get all skill names
                for demo_id in range(len(demos)):
                    sg_info = f[f'data/demo_{demo_id}/sg_info']
                    for skill_name in sg_info.keys():
                        for skill_key in primitive_kws:
                            if skill_key in skill_name:
                                interested_skills.add(skill_name)
                                skill_info = sg_info[skill_name]
                                interested_objs.add(skill_info['related_objs'][0].decode('utf-8'))
                                break
                        else:
                            ## if no interested skill found, skip this skill
                            continue

                skill_name_to_emb = get_embs_without_saving(list(interested_skills), cache_dir=cache_dir)
                skill_embs_all_tasks.update(skill_name_to_emb)
                self.involved_skill_names = self.involved_skill_names.union(interested_skills)

                for demo_id in range(len(demos)):
                    sg_info = f[f'data/demo_{demo_id}/sg_info']
                
                    obs_grp = f[f'data/demo_{demo_id}/obs']
                    rbt_states = get_rbt_states(obs_grp, robot_names)
                    obj_pcds = get_pc_instances(obs_grp, interested_objs)
                    action_arr = f[f'data/demo_{demo_id}/actions'][()]
                    rbt_action = get_rbt_actions(action_arr, robot_names)

                    for _ in range(traj_nums):
                        # Create separate data slices for each skill name
                        for skill_name, skill_info in sg_info.items():

                            ## only use bimanual skills
                            if 'bimanual' in skill_name:
                                # data_slice = self.get_dataslice_bimanual_kp(skill_info, skill_name, obj_pcds, traj_len, rbt_states, rbt_action, task_name)
                                continue
                            elif skill_name in interested_skills:
                                data_slice = self.get_dataslice_unimanual(skill_info, skill_name, skill_key, cfg, traj_len, obj_pcds, rbt_states, rbt_action, task_name)
                            else:
                                continue
                                   
                            data_list.append(data_slice)
        

        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 files!')

        # cfg.skill_names = list(self.involved_skill_names)
        # print(f'Involved skill names: {cfg.skill_names}')
        ## obtain skill name embedding
        cache_name = f'{cfg.dataset_type}_skill_name_to_emb.npy'
        save_embs(skill_embs_all_tasks, cache_dir=cache_dir, cache_name=cache_name)

        task_emb_dict = get_embs_without_saving(list(involved_tasks), cache_dir=cache_dir)
        self.statistics['task_emb_dict'] = task_emb_dict
        self.statistics['skill_embs_all_tasks'] = skill_embs_all_tasks
        self.statistics['matched_action_sgs'] = matched_action_sgs
        return data_list

    ## note that this dual_manual dataset cannot merge with unimanual dataset
    def get_dataslice_bimanual_kp(self, skill_info, skill_name,  obj_pcds, traj_len, rbt_states, rbt_action, task_name):
        data_slice = {}

        pre_sg = get_sg(skill_info, 'pre_sg')
        pre_idx_list = pre_sg.graph['idx_list']

        ## obtain the pc at the first several frame (10 frames)
        observation_idx = 10
        ## normalize method 1
        # init_pc = []
        # for obj_name in obj_pcds.keys():
        #     init_pc.append(obj_pcds[obj_name][10][:, :3])
        # init_pc = np.concatenate(init_pc, axis=0)
        # init_pc_n, init_pc_offset = centralize_downsample(init_pc, self.pc_shape, obj_centric = self.is_obj_centric, add_bottom = self.is_add_bottom, method = self.downsample_method, debug_visualize=False)

        ## normalize method 2
        part_pc_shape= (self.pc_shape[0]//2, 3)
        related_pc_dict = {obj_name: obj_pcds[obj_name][observation_idx]for obj_name in obj_pcds.keys()}
        init_pc_n, init_pc_offset = combined_pc_instances_and_offset(related_pc_dict, part_pc_shape, self.is_obj_centric, self.is_add_bottom, self.downsample_method)
  

        ## random select one eefpose at the switch point
        ## TODO: we can also learn the bimanual traj

        random_switch_id = np.random.choice(pre_idx_list)
        switch_xyz_left = rbt_states['robot0_eef_pos'][random_switch_id]
        switch_xyz_right = rbt_states['robot1_eef_pos'][random_switch_id]
        switch_quat_left = rbt_states['robot0_eef_quat'][random_switch_id]
        switch_quat_right = rbt_states['robot1_eef_quat'][random_switch_id]
        left_eef_trans = compose_transformation(switch_xyz_left, switch_quat_left)
        left_eef_trans = centralize_grasp(left_eef_trans, init_pc_offset)
        right_eef_trans = compose_transformation(switch_xyz_right, switch_quat_right)
        right_eef_trans = centralize_grasp(right_eef_trans, init_pc_offset)
        
        ## make the bikp compatible to the unimanual dataset
        rep_times = traj_len // 2
        left_eef_trans_rep = np.tile(left_eef_trans, (rep_times, 1, 1))
        right_eef_trans_rep = np.tile(right_eef_trans, (rep_times, 1, 1))
        pre_dual_eef = np.concatenate([left_eef_trans_rep, right_eef_trans_rep], axis=0)

        ## gripper action, won't be used
        gripper_left = rbt_action['robot0'][random_switch_id]
        gripper_right = rbt_action['robot1'][random_switch_id]
        gripper_left_rep = np.tile(gripper_left, (rep_times, 1, 1))
        gripper_right_rep = np.tile(gripper_right, (rep_times, 1, 1))
        gripper_list = np.concatenate([gripper_left_rep, gripper_right_rep], axis=0)

        data_slice['eefpos'] = torch.tensor(pre_dual_eef).to(torch.float32)
        data_slice['gripper'] = torch.tensor(gripper_list).to(torch.float32)
        data_slice['pc'] = torch.tensor(init_pc_n).to(torch.float32).reshape(1, -1, 3)
        data_slice['skill_name'] = str_to_ascii_tensor(skill_name)
        data_slice['task_name'] = str_to_ascii_tensor(task_name)

        return data_slice

    def get_obj_pc_tensor(self, obj_pc_list, observation_idx, num_points):
        obj_pc = obj_pc_list[observation_idx][:, :3]
        
        obj_pc_n, obj_offset = centralize_downsample(obj_pc, self.pc_shape, obj_centric = self.is_obj_centric, add_bottom = self.is_add_bottom, method = self.downsample_method, debug_visualize=False)
        obj_pc_tensor = torch.tensor(obj_pc_n).unsqueeze(0).to(torch.float32).reshape(1, num_points, 3)
        return obj_pc_tensor, obj_offset

    def decide_in_hand_obj(self,  cur_sg, obj_pcds, obj_name):
        for edge in cur_sg.edges:
            entities = set(edge)
            if obj_name in entities:
                other_entity = list(entities - {obj_name})
                if other_entity[0] in obj_pcds.keys():
                    in_hand_obj_name = other_entity[0]
                    break
        else:
            in_hand_obj_name = None

        return in_hand_obj_name

    def get_dataslice_unimanual(self, skill_info, skill_name, skill_key, cfg, traj_len,  obj_pcds, rbt_states, rbt_action,  task_name):
        data_slice = {}

        # pre_sg = get_sg(skill_info, 'pre_sg')
        cur_sg = get_sg(skill_info, 'cur_sg')
        # eff_sg = get_sg(skill_info, 'eff_sg')
        ## diverse pc input
        pre_start_idx = 0
        obj_pc_idx = np.random.randint(pre_start_idx, pre_start_idx + 5)

        # obj_name = skill_condition_objs[skill_name]
        related_objs = skill_info['related_objs'][()]
        obj_name = related_objs[0].decode('utf-8')
        rbt_name = skill_info['related_rbts'][0].decode('utf-8')
        idx_list = skill_info['extended_ids'][()]
        essential_ids = skill_info['essential_ids'][()]

        obj_pc_tensor, obj_offset = self.get_obj_pc_tensor(obj_pcds[obj_name], obj_pc_idx, cfg.num_points)

        ####### if more than one objs, get another obj pc

        in_hand_obj_name = self.decide_in_hand_obj(cur_sg, obj_pcds, obj_name)

        if in_hand_obj_name is not None:
            in_hand_obj_pc_tensor, _ = self.get_obj_pc_tensor(obj_pcds[in_hand_obj_name], obj_pc_idx, cfg.num_points)
            # in_hand_mask_tensor = torch.tensor([True], dtype=torch.bool)
        else:            
            in_hand_obj_pc_tensor = obj_pc_tensor.clone()
            # in_hand_mask_tensor = torch.tensor([False], dtype=torch.bool)
        #########
        
        if cfg.choose_id_method == "rdp":   
            chosen_ids = choose_ids_rdp(rbt_states[f'{rbt_name}_eef_pos'], traj_len, idx_list, essential_ids = essential_ids)
        else:
            chosen_ids = choose_ids(traj_len, idx_list, essential_ids, skill_key)
        eef_pos_list = rbt_states[f'{rbt_name}_eef_pos'][chosen_ids]
        eef_quat_list = rbt_states[f'{rbt_name}_eef_quat'][chosen_ids]
        eef_pos_list = list(map(compose_transformation, eef_pos_list, eef_quat_list))
        normalized_eef_pos_list = list(map(centralize_grasp, eef_pos_list, [obj_offset]*traj_len))
        normalized_eef_pos_tensor = torch.tensor(normalized_eef_pos_list).to(torch.float32).reshape(traj_len, 4, 4) 

        gripper_list = rbt_action[rbt_name][chosen_ids]
        
        ## input
        data_slice['pc'] = obj_pc_tensor
        data_slice['in_hand_pc'] = in_hand_obj_pc_tensor
        # data_slice['in_hand_mask'] = in_hand_mask_tensor
        # data_slice['skill_name_emb'] = skill_name_to_emb[skill_name]
        ## output
        data_slice['eefpos'] = normalized_eef_pos_tensor
        data_slice['gripper'] = torch.tensor(gripper_list).to(torch.float32).reshape(traj_len, 1, 1)
        data_slice['skill_name'] = str_to_ascii_tensor(skill_name)
        data_slice['task_name'] = str_to_ascii_tensor(task_name)
        ## note: if rotation, then the min xy and max xy will be same. So we need mean instead of min/max
        if cfg.rot_aug:
            data_slice = rotate_dataslice(data_slice)

        return data_slice



    def get_normalizer_and_statistics(self, data_list, mode = 'unimanual'):
        normalizer = LinearNormalizer()

        if mode == 'bimanual':
            jpose_arr = np.concatenate([data['jpose'].reshape(2, -1) for data in data_list], axis=0)
            jpose_stats = to_torch_stats(jpose_arr.reshape(-1, jpose_arr.shape[-1]))
            normalizer['jpose'] = get_torch_range_symmetric_normalizer_from_stat(jpose_stats)
            return normalizer

        ### normalize pc
        pc_arr = np.concatenate([data['pc'] for data in data_list], axis=0)
        pcd_stats = to_torch_stats(pc_arr.reshape(-1, pc_arr.shape[-1]))

        normalizer['pc'] = get_torch_range_symmetric_normalizer_from_stat(pcd_stats)


        ## normalize eefpos. first convert to 3vec or 4pts
        eef_pos_arr = np.concatenate([data['eefpos'] for data in data_list], axis=0)
        eef_pos_torch = torch.tensor(eef_pos_arr).to(torch.float32)

        if self.eef_representation == '3vec':
            eef_xyz_raw, _, _ = convert_trans_to_vec(eef_pos_torch.reshape(-1, 1, 4, 4))
            eef_xyz_np = eef_xyz_raw.detach().cpu().numpy()
            eef_stats = to_torch_stats(eef_xyz_np.reshape(-1, eef_xyz_np.shape[-1]))
        elif self.eef_representation == '4pts':
            original_gripper_pcd = np.array(self.original_gripper_pcd)
            eef_4pts_raw = convert_trans_to_4pts(eef_pos_torch.reshape(-1, 1, 4, 4), original_gripper_pcd)
            eef_4pts_np = eef_4pts_raw.detach().cpu().numpy()
            eef_stats = to_torch_stats(eef_4pts_np.reshape(-1, eef_4pts_np.shape[-1]))
        ## actually not symmetric. confusing name.
        normalizer['eefpos'] = get_torch_range_symmetric_normalizer_from_stat(eef_stats)
        
        ## normalize gripper
        gripper_arr = np.concatenate([data['gripper'] for data in data_list], axis=0)
        gripper_stats = to_torch_stats(gripper_arr.reshape(-1, gripper_arr.shape[-1]))
        normalizer['gripper'] = get_torch_range_symmetric_normalizer_from_stat(gripper_stats)

        ## set_scale. 
        pc_scale = self.get_pc_scale(pc_arr, eef_stats["max"].max())
        self.statistics['pc_scale'] = pc_scale

        if 'in_hand_pc' in data_list[0]:
            in_hand_pc_arr = np.concatenate([data['in_hand_pc'] for data in data_list], axis=0)
            in_hand_pcd_stats = to_torch_stats(in_hand_pc_arr.reshape(-1, in_hand_pc_arr.shape[-1]))
            normalizer['in_hand_pc'] = get_torch_range_symmetric_normalizer_from_stat(in_hand_pcd_stats)

            # in_hand_pc_scale = self.get_pc_scale(in_hand_pc_arr, eef_stats["max"].max())
            # self.statistics['in_hand_pc_scale'] = in_hand_pc_scale

        return normalizer

    
    def get_pc_scale(self, pc_data, ac_scale):
        """
        pc_data: (N_demos*demo_len, num_points, 3)
        """
        centroid = pc_data.mean(axis=1, keepdims=True)
        centered_pc = pc_data - centroid
        pc_scale = np.linalg.norm(centered_pc, axis=-1).mean()
        normed_pc_scale = pc_scale / ac_scale
        return normed_pc_scale
    
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

                        
                        
