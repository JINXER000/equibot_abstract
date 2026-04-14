import os
import h5py
import hydra
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from equibot.policies.utils.misc import (
    EQUIBOT_PATH, 
    compose_transformation, 
    centralize_downsample, 
    centralize_grasp, 
    choose_ids, 
    choose_ids_rdp, 
    rotate_dataslice, 
    str_to_ascii_tensor,
    convert_trans_to_vec,
    convert_trans_to_4pts
)

from equibot.policies.utils.lan_utils import get_embs_without_saving, save_embs
from equibot.policies.utils.normalize_utils import to_torch_stats, get_torch_range_symmetric_normalizer_from_stat
from equibot.policies.utils.normalizer import LinearNormalizer



class RealAlohaDataset(Dataset):
    """
    Dataset for real ALOHA robot data with unimanual trajectory training.
    
    Expected HDF5 structure (per episode):
    - cam_high: shape=(T, H, W, 3) - camera images
    - <obj_name>/  (e.g., 'cup')
        - start_pc: shape=(N, 3) - point cloud at start
        - start_colors: shape=(N, 3) - RGB colors for start_pc
        - end_pc: shape=(N, 3) - point cloud at effect (optional)
        - end_colors: shape=(N, 3) - RGB colors for end_pc (optional)
        - grasp_ids: shape=(K,) - indices for grasp frames
        - grasp_poses: shape=(K, 4, 4) - grasp poses
        - holding_ids: shape=(M,) - indices during holding
        - joint_poses: shape=(M, 14) - joint poses (7 per arm)
        - release_ids: shape=(L,) - indices for release frames
        - release_poses: shape=(L, 4, 4) - release poses
        - eff_holding_ids: shape=(E,) - effect holding indices (optional)
    """
    
    def __init__(self, cfg, mode, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        super().__init__()
        self.mode = mode
        self.dir_name = cfg.path
        self.root = self.dir_name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.composed_inference = False

        self.use_pc_color = cfg.use_pc_color if hasattr(cfg, 'use_pc_color') else False
        pc_channels = 6 if self.use_pc_color else 3
        self.pc_shape = (cfg.num_points, pc_channels)
        
        self.is_obj_centric = cfg.is_obj_centric
        self.is_add_bottom = cfg.is_add_bottom if hasattr(cfg, 'is_add_bottom') else False
        self.downsample_method = cfg.downsample_method if hasattr(cfg, 'downsample_method') else 'fps'

        self.num_eef = cfg.num_eef if hasattr(cfg, 'num_eef') else 1
        self.dof = cfg.dof if hasattr(cfg, 'dof') else 7
        self.dataset_type = cfg.dataset_type

        self.eef_representation = cfg.eef_representation if hasattr(cfg, 'eef_representation') else '3vec'
        self.original_gripper_pcd = np.array(cfg.original_gripper_pcd) if hasattr(cfg, 'original_gripper_pcd') else None

        self.statistics = {}

        if mode == 'train':
            print('Processing real ALOHA dataset...')
            self.process_select(cfg, **kwargs)
            self.skill_names = list(self.statistics.get('skill_embs_all_tasks', {}).keys())
            self.task_names = list(self.statistics.get('task_emb_dict', {}).keys())
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

    def get_pc_of_phase(self, obj_grp, phase = 'start'):

        # Get point cloud data
        pc = obj_grp[f'{phase}_pc'][()]
        
        # Handle color data
        if self.use_pc_color and f'{phase}_colors' in obj_grp:
            pc_colors = obj_grp[f'{phase}_colors'][()]
            # Combine xyz and colors
            obj_pc_raw = np.concatenate([pc, pc_colors], axis=-1)
        else:
            obj_pc_raw = pc

        return obj_pc_raw


    def process_select(self, cfg, **kwargs):
        if self.dataset_type == 'real_aloha_traj':
            # Unimanual trajectory dataset (pc + eef pose sequence)
            self.data = self.process_real_aloha_traj(cfg, **kwargs)
            self.normalizer = self.get_normalizer_and_statistics(self.data, mode='unimanual')
        elif self.dataset_type == 'real_aloha_biop_jpose':
            # Bimanual joint-pose dataset (single joint configuration per sample)
            self.data = self.process_real_aloha_biop_jpose(cfg, **kwargs)
            self.normalizer = self.get_normalizer_and_statistics(self.data, mode='bimanual')
        else:
            raise NotImplementedError(f'Dataset type {self.dataset_type} not implemented!')

    def process_real_aloha_traj(self, cfg, **kwargs):
        """
        Process real ALOHA trajectory data for unimanual skill learning.
        
        This follows the pattern from per_skill_dataset.py but adapted for the
        real ALOHA data format with simpler skill structure.
        """
        print('Processing real ALOHA hdf5 dataset...')
        data_list = []
        raw_files = self.raw_file_names
        traj_len = cfg.pred_horizon
        aug_traj_nums = cfg.aug_traj_nums if hasattr(cfg, 'aug_traj_nums') else 64
        
        cache_dir = os.path.join(EQUIBOT_PATH, cfg.embedding_cache_dir) if hasattr(cfg, 'embedding_cache_dir') else None
        
        # Get task name from config or directory name
        task_name = cfg.task_name 
        
        # Get uniskills from config
        primitive_kws = cfg.uniskills if hasattr(cfg, 'uniskills') else ['grasp', 'place']
        
        self.involved_skill_names = set()
        skill_embs_all_tasks = {}
        
        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]
            if not file_name.endswith('.hdf5'):
                continue
            
            hdf5_path = os.path.join(self.root, 'raw', file_name)
            
            with h5py.File(hdf5_path, 'r') as f:
                # Find object groups (exclude camera data and non-group items)
                obj_names = [key for key in f.keys() 
                             if isinstance(f[key], h5py.Group)]
                
                for obj_name in obj_names:
                    obj_grp = f[obj_name]
                    
                    # Get point cloud data
                    start_pc = self.get_pc_of_phase(obj_grp, phase = 'start')
                    
                    # Get grasp poses and indices
                    grasp_poses = obj_grp['grasp_poses'][()]
                    grasp_ids = obj_grp['grasp_ids'][()]
                    grasp_gripper_actions = obj_grp['grasp_gripper_actions'][()]
                    
                    # Get release poses if available (for place skill)
                    has_release = 'release_poses' in obj_grp
                    if has_release:
                        release_poses = obj_grp['release_poses'][()]
                        release_ids = obj_grp['release_ids'][()]
                        end_pc = self.get_pc_of_phase(obj_grp, phase = 'end')
                        release_gripper_actions = obj_grp['release_gripper_actions'][()]

                    
                    # Create data samples for grasp skill
                    for _ in range(aug_traj_nums):
                        for primitive_kw in primitive_kws:
                            # Process grasp skill
                            if primitive_kw == 'grasp' and len(grasp_poses) > 0:
                                skill_name = f'grasp_{obj_name}'
                                obj_pc_raw = start_pc
                                eef_poses = grasp_poses
                                pose_ids = grasp_ids
                                gripper_actions = grasp_gripper_actions
                            elif primitive_kw == 'place' and has_release and len(release_poses) > 0:
                                skill_name = f'place_{obj_name}'
                                obj_pc_raw = end_pc
                                eef_poses = release_poses
                                pose_ids = release_ids
                                gripper_actions = release_gripper_actions
                            else:
                                raise ValueError(f'No valid skill found for {obj_name} in {file_name}')

                            data_slice = self.get_dataslice_unimanual_real(
                                obj_pc_raw=obj_pc_raw,
                                eef_poses=eef_poses,
                                pose_ids=pose_ids,
                                skill_name=skill_name,
                                task_name=task_name,
                                cfg=cfg,
                                traj_len=traj_len,
                                gripper_action= gripper_actions
                            )
                            if data_slice is not None:
                                data_list.append(data_slice)
                                self.involved_skill_names.add(skill_name)
        
        # Save processed data
        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print(f'Processed all hdf5 files! Total samples: {len(data_list)}')
        
        # Get skill name embeddings
        if cache_dir is not None and len(self.involved_skill_names) > 0:
            skill_name_to_emb = get_embs_without_saving(list(self.involved_skill_names), cache_dir=cache_dir)
            skill_embs_all_tasks.update(skill_name_to_emb)
            cache_name = f'{cfg.dataset_type}_skill_name_to_emb.npy'
            save_embs(skill_embs_all_tasks, cache_dir=cache_dir, cache_name=cache_name)
        
        # Get task embedding
        task_emb_dict = {}
        if  task_name is not None:
            task_emb_dict = get_embs_without_saving([task_name], cache_dir=cache_dir)
        else:
            ## task emb is the same as skill emb
            task_emb_dict = skill_embs_all_tasks
            
        self.statistics['task_emb_dict'] = task_emb_dict

        self.statistics['skill_embs_all_tasks'] = skill_embs_all_tasks
        
        return data_list

    def process_real_aloha_biop_jpose(self, cfg, **kwargs):
        """
        Process real ALOHA data into a bimanual joint-pose (biop) dataset.

        This mirrors `process_per_biop` in `per_skill_dataset.py` but adapts
        to the simpler real ALOHA HDF5 format, which stores joint poses and
        holding indices directly under each object group.
        """
        print('Processing real ALOHA bimanual joint-pose dataset...')
        data_list = []
        raw_files = self.raw_file_names

        aug_traj_nums = cfg.aug_traj_nums if hasattr(cfg, 'aug_traj_nums') else 64
        cache_dir = os.path.join(EQUIBOT_PATH, cfg.embedding_cache_dir) if hasattr(cfg, 'embedding_cache_dir') else None

        # Task name is shared across all files for this dataset
        task_name = cfg.task_name

        self.involved_skill_names = set()
        skill_embs_all_tasks = {}

        for file_id in range(len(raw_files)):
            file_name = raw_files[file_id]
            if not file_name.endswith('.hdf5'):
                continue

            hdf5_path = os.path.join(self.root, 'raw', file_name)
            with h5py.File(hdf5_path, 'r') as f:
                # Find object groups (exclude camera data and non-group items)
                obj_names = [key for key in f.keys() 
                             if isinstance(f[key], h5py.Group)]

                for obj_name in obj_names:
                    obj_grp = f[obj_name]

                    # Require joint poses and holding ids for bimanual pose sampling
                    if 'joint_poses' not in obj_grp or 'holding_ids' not in obj_grp:
                        continue

                    joint_poses = obj_grp['joint_poses'][()]  # shape (M, 14) for dual-arm ALOHA
                    holding_ids = obj_grp['holding_ids'][()]

                    if len(joint_poses) == 0 or len(holding_ids) == 0:
                        continue

                    # Skill name is per-object bimanual holding
                    skill_name = f'bimanual_hold_{obj_name}'

                    for _ in range(aug_traj_nums):
                        data_slice = self.get_dataslice_bimanual_jpose(
                            joint_poses=joint_poses,
                            skill_name=skill_name,
                            task_name=task_name,
                        )
                        if data_slice is not None:
                            data_list.append(data_slice)
                            self.involved_skill_names.add(skill_name)

        # Save processed data
        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print(f'Processed all hdf5 files for bimanual jpose! Total samples: {len(data_list)}')

        # Get skill name embeddings (optional)
        if cache_dir is not None and len(self.involved_skill_names) > 0:
            skill_name_to_emb = get_embs_without_saving(list(self.involved_skill_names), cache_dir=cache_dir)
            skill_embs_all_tasks.update(skill_name_to_emb)
            cache_name = f'{self.dataset_type}_skill_name_to_emb.npy'
            save_embs(skill_embs_all_tasks, cache_dir=cache_dir, cache_name=cache_name)

        # Get task embedding
        task_emb_dict = {}
        if task_name is not None:
            task_emb_dict = get_embs_without_saving([task_name], cache_dir=cache_dir)
        else:
            task_emb_dict = skill_embs_all_tasks

        self.statistics['task_emb_dict'] = task_emb_dict
        self.statistics['skill_embs_all_tasks'] = skill_embs_all_tasks

        return data_list

    def get_dataslice_bimanual_jpose(self, joint_poses,  skill_name, task_name):
        """
        Create a data slice for bimanual joint-pose learning from real ALOHA data.

        This is conceptually similar to `get_dataslice_bimanual_jpose` in
        `per_skill_dataset.py` but uses joint poses and holding indices directly,
        without relying on scene graphs or robot state dictionaries.
        """
        valid_ids = range(len(joint_poses)//6, len(joint_poses) // 3)
        # valid_ids = range(len(joint_poses)//6*5, len(joint_poses))

        # Randomly select one holding index and use the corresponding joint pose
        qtraj_index = np.random.choice(valid_ids)
        selected_jpose = joint_poses[qtraj_index].astype(np.float32)  # shape (14,)

        data_slice_bi = {}
        data_slice_bi['jpose'] = selected_jpose
        data_slice_bi['skill_name'] = str_to_ascii_tensor(skill_name)
        data_slice_bi['task_name'] = str_to_ascii_tensor(task_name) if task_name is not None else data_slice_bi['skill_name']

        return data_slice_bi


    def get_dataslice_unimanual_real(self, obj_pc_raw, eef_poses, pose_ids, skill_name, task_name, cfg, traj_len, gripper_action):
        """
        Create a data slice for unimanual skill learning from real ALOHA data.
        
        This follows the structure from per_skill_dataset.get_dataslice_unimanual but
        adapted for the simpler real ALOHA data format.
        """
        data_slice = {}
        
        if len(eef_poses) == 0:
            return None
        
        # Downsample and centralize point cloud
        obj_pc_n, obj_offset = centralize_downsample(
            obj_pc_raw,
            self.pc_shape,
            obj_centric=self.is_obj_centric,
            add_bottom=self.is_add_bottom,
            method=self.downsample_method,
            debug_visualize=False
        )
        obj_pc_tensor = torch.tensor(obj_pc_n).unsqueeze(0).to(torch.float32)
        
        # Sample trajectory indices based on gripper action values
        chosen_ids = self._sample_trajectory_indices(len(eef_poses), traj_len, gripper_action)
        
        # Get eef poses and normalize
        eef_pos_list = eef_poses[chosen_ids]
        normalized_eef_pos_list = np.array([centralize_grasp(eef_pos.copy(), obj_offset) for eef_pos in eef_pos_list])
        normalized_eef_pos_tensor = torch.tensor(normalized_eef_pos_list).to(torch.float32).reshape(traj_len, 4, 4)
        
        # Create gripper actions (same action for all steps in this skill)
        # gripper_list = np.full((traj_len, 1, 1), gripper_action, dtype=np.float32)
        gripper_list = gripper_action[chosen_ids]
        
        # Build data slice
        data_slice['pc'] = obj_pc_tensor
        data_slice['in_hand_pc'] = obj_pc_tensor.clone()  # For compatibility with per_skill format
        data_slice['eefpos'] = normalized_eef_pos_tensor
        data_slice['gripper'] = torch.tensor(gripper_list).to(torch.float32).reshape(traj_len, 1, 1)
        data_slice['skill_name'] = str_to_ascii_tensor(skill_name)
        ## if task_name is not distinctive, use skill_name instead
        data_slice['task_name'] = str_to_ascii_tensor(task_name) if task_name is not None else data_slice['skill_name'] 
        
        # Apply rotation augmentation if enabled
        if hasattr(cfg, 'rot_aug') and cfg.rot_aug:
            data_slice = rotate_dataslice(data_slice)
        
        return data_slice

    def _sample_trajectory_indices(self, total_len, traj_len, gripper_action=None):
        """Sample trajectory indices based on gripper action values.
        
        Priority order:
        1. First (init) and last (end) indices
        2. Max gripper value index
        3. Min gripper value index
        4. Intermediate indices based on gripper values (e.g., midpoint between min and max)
        """
        if total_len <= traj_len:
            return np.linspace(0, total_len - 1, traj_len).astype(int)
        
        # If no gripper action provided, fall back to linear spacing
        if gripper_action is None:
            return np.linspace(0, total_len - 1, traj_len).astype(int)
        
        # Flatten gripper_action if needed
        gripper_values = np.array(gripper_action).flatten()
        
        # Start with init (0) and end (total_len - 1) indices
        chosen_set = {0, total_len - 1}
        
        # Find max and min gripper indices (excluding first and last)
        middle_range = np.arange(1, total_len - 1)
        if len(middle_range) > 0:
            middle_gripper = gripper_values[middle_range]
            max_idx = middle_range[np.argmax(middle_gripper)]
            min_idx = middle_range[np.argmin(middle_gripper)]
            chosen_set.add(max_idx)
            chosen_set.add(min_idx)
        
        # Fill remaining slots by uniformly sampling from available indices
        remaining_slots = traj_len - len(chosen_set)
        if remaining_slots > 0:
            available_indices = np.array([i for i in range(total_len) if i not in chosen_set])
            if len(available_indices) > 0:
                sampled = np.random.choice(available_indices, size=min(remaining_slots, len(available_indices)), replace=False)
                chosen_set.update(sampled)
        
        # Convert to sorted array
        chosen_ids = np.sort(np.array(list(chosen_set)))
        return chosen_ids.astype(int)

    def get_normalizer_and_statistics(self, data_list, mode='unimanual'):
        """Compute normalizer statistics from data."""
        normalizer = LinearNormalizer()

        if len(data_list) == 0:
            return normalizer

        # Bimanual joint-pose only: normalize joint positions and return
        if mode == 'bimanual':
            jpose_arr = np.stack([data['jpose'] for data in data_list], axis=0)
            jpose_stats = to_torch_stats(jpose_arr.reshape(-1, jpose_arr.shape[-1]))
            normalizer['jpose'] = get_torch_range_symmetric_normalizer_from_stat(jpose_stats)
            return normalizer

        # Unimanual trajectory: normalize pc, eefpos, gripper, in_hand_pc
        pc_arr = np.concatenate([data['pc'] for data in data_list], axis=0)

        if self.use_pc_color and pc_arr.shape[-1] == 6:
            pc_xyz = pc_arr[..., :3]
        elif pc_arr.shape[-1] == 3:
            pc_xyz = pc_arr
        else:
            raise ValueError(f"Invalid pc shape: {pc_arr.shape}")

        pcd_stats = to_torch_stats(pc_xyz.reshape(-1, 3))
        normalizer['pc'] = get_torch_range_symmetric_normalizer_from_stat(pcd_stats)

        # Normalize eefpos
        eef_pos_arr = np.concatenate([data['eefpos'] for data in data_list], axis=0)
        eef_pos_torch = torch.tensor(eef_pos_arr).to(torch.float32)

        if self.eef_representation == '3vec':
            eef_xyz_raw, _, _ = convert_trans_to_vec(eef_pos_torch.reshape(-1, 1, 4, 4))
            eef_xyz_np = eef_xyz_raw.detach().cpu().numpy()
            eef_stats = to_torch_stats(eef_xyz_np.reshape(-1, eef_xyz_np.shape[-1]))
        elif self.eef_representation == '4pts':
            original_gripper_pcd = self.original_gripper_pcd
            eef_4pts_raw = convert_trans_to_4pts(eef_pos_torch.reshape(-1, 1, 4, 4), original_gripper_pcd)
            eef_4pts_np = eef_4pts_raw.detach().cpu().numpy()
            eef_stats = to_torch_stats(eef_4pts_np.reshape(-1, eef_4pts_np.shape[-1]))
        else:
            # Default: use xyz position stats
            eef_xyz_raw, _, _ = convert_trans_to_vec(eef_pos_torch.reshape(-1, 1, 4, 4))
            eef_xyz_np = eef_xyz_raw.detach().cpu().numpy()
            eef_stats = to_torch_stats(eef_xyz_np.reshape(-1, eef_xyz_np.shape[-1]))

        normalizer['eefpos'] = get_torch_range_symmetric_normalizer_from_stat(eef_stats)

        # Normalize gripper
        gripper_arr = np.concatenate([data['gripper'] for data in data_list], axis=0)
        gripper_stats = to_torch_stats(gripper_arr.reshape(-1, gripper_arr.shape[-1]))
        normalizer['gripper'] = get_torch_range_symmetric_normalizer_from_stat(gripper_stats)

        # Compute pc_scale
        pc_arr_for_scale = pc_arr[..., :3] if pc_arr.shape[-1] == 6 else pc_arr
        pc_scale = self.get_pc_scale(pc_arr_for_scale, eef_stats["max"].max())
        self.statistics['pc_scale'] = pc_scale

        # Normalize in_hand_pc (same as pc for real aloha)
        if 'in_hand_pc' in data_list[0]:
            in_hand_pc_arr = np.concatenate([data['in_hand_pc'] for data in data_list], axis=0)
            if self.use_pc_color and in_hand_pc_arr.shape[-1] == 6:
                in_hand_pc_xyz = in_hand_pc_arr[..., :3]
            elif in_hand_pc_arr.shape[-1] == 3:
                in_hand_pc_xyz = in_hand_pc_arr
            else:
                raise ValueError(f"Invalid in_hand_pc shape: {in_hand_pc_arr.shape}")

            in_hand_pcd_stats = to_torch_stats(in_hand_pc_xyz.reshape(-1, 3))
            normalizer['in_hand_pc'] = get_torch_range_symmetric_normalizer_from_stat(in_hand_pcd_stats)

        return normalizer

    def get_pc_scale(self, pc_data, ac_scale):
        """Compute point cloud scale relative to action scale."""
        centroid = pc_data.mean(axis=1, keepdims=True)
        centered_pc = pc_data - centroid
        pc_scale = np.linalg.norm(centered_pc, axis=-1).mean()
        normed_pc_scale = pc_scale / ac_scale
        return normed_pc_scale


@hydra.main(config_path=os.path.join(EQUIBOT_PATH, "equibot/policies/configs"), config_name="sdp_per_skill")
def main(cfg):
    """Test the dataset loading."""
    import sys
    sys.path.append('/home/user/yzchen_ws/TAMP-ubuntu22/pddlstream_aloha')
    
    # Override config for testing
    cfg.data.dataset.dataset_type = 'real_aloha_traj'
    cfg.data.dataset.path = os.path.join(EQUIBOT_PATH, 'data/handoff_cup/')
    cfg.data.dataset.task_name = 'handoff_cup'
    
    test_dataset = RealAlohaDataset(cfg.data.dataset, "train")
    print(f"Dataset size: {len(test_dataset)}")
    print(f"Skill names: {test_dataset.skill_names}")
    print(f"Task names: {test_dataset.task_names}")
    
    # Test loading a sample
    if len(test_dataset) > 0:
        sample = test_dataset[0]
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"  {key}: {type(val)}")


if __name__ == '__main__':
    main()
