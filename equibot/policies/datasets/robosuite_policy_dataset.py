import os
import glob
import time
import h5py
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
import hydra
import torch
import networkx as nx
from equibot.policies.utils.misc import rotate_around_z, EQUIBOT_PATH, compose_transformation,\
matrix_to_rotation_6d


class RobosuitePolicyDataset(Dataset):
    def __init__(self, cfg, mode) -> None:
        super().__init__()

        self.mode = mode
        self.dof = cfg["dof"]
        self.num_eef = cfg["num_eef"]
        self.eef_dim = cfg["eef_dim"]
        self.num_points = cfg["num_points"]
        self.num_augment = cfg["num_augment"]
        self.aug_keep_original = cfg["aug_keep_original"]
        self.aug_scale_low = cfg["aug_scale_low"]
        self.aug_scale_high = cfg["aug_scale_high"]
        self.aug_scale_aspect_limit = cfg["aug_scale_aspect_limit"]
        self.aug_scale_pos = cfg["aug_scale_pos"]
        self.aug_scale_rot = cfg["aug_scale_rot"]
        self.aug_center = np.array(cfg["aug_center"])
        self.same_aug_per_sample = cfg["same_aug_per_sample"]
        self.aug_zero_z_offset = cfg["aug_zero_z_offset"]
        self.reduce_horizon_dim = cfg["reduce_horizon_dim"]
        self.shuffle_pc = cfg["shuffle_pc"]
        self.min_demo_length = cfg["min_demo_length"]
        
        if "latency" in cfg:
            self.state_latency = cfg["latency"]["state"]
            self.state_latency_random = cfg["latency"]["random"]
        else:
            self.state_latency = 0
        if "obs_horizon" in cfg:
            self.obs_horizon = cfg["obs_horizon"]
            self.pred_horizon = cfg["pred_horizon"]
        else:
            self.obs_horizon = 1
            self.pred_horizon = 1

        if mode == 'inference':
            print('This is a dummy dataset for inference, no data loading')
            return

        self.root = cfg["path"]
        self.data_dir = os.path.join(self.root, "raw")
        
        # Get HDF5 files
        self.hdf5_files = [f for f in os.listdir(self.data_dir) if 'hdf5' in f]
        
        if "num_demos" in cfg and cfg["num_demos"] < len(self.hdf5_files):
            print(f"[dataset.py] Filtering demos to {cfg['num_demos']} demos")
            self.hdf5_files = self.hdf5_files[:cfg["num_demos"]]
        else:
            print("[dataset.py] Using all demos")

        # Build episode metadata and timestep indices
        self.timestep_indices = []  # List of (file_idx, timestep_idx) tuples
        # self.ep_length_dict = {}
        self.ep_metadata = {}  # Store metadata for each episode
        

        
        # Pre-load and cache data
        self._init_cache()


    def __len__(self):
        return len(self.timestep_indices)

    def _init_cache(self):
        """Pre-load and cache all data from HDF5 files"""
        self.cache = {}
        for file_idx, file_name in enumerate(tqdm(self.hdf5_files, desc="Caching data")):
            hdf5_path = os.path.join(self.data_dir, file_name)

            with h5py.File(hdf5_path, 'r') as f:
                demo_id = file_name.split('_')[-2]
                ep_key = f"{file_name}"
               
                self.cache[ep_key] = {}

              # Get absolute actions and gripper info
                # abs_actions = f[f'data/abs_actions'][()]
                # abs_actions = abs_actions.reshape(*abs_actions.shape[:1], -1, 7)
                # gripper_array = abs_actions[..., [-1]]

                ## NOTE: in action_dict, robot1 --> left, robot0 --> right
                left_gripper_actions = f[f'data/demo_{demo_id}/action_dict/left_gripper'][()]
                right_gripper_actions = f[f'data/demo_{demo_id}/action_dict/right_gripper'][()]
                gripper_array = np.concatenate([right_gripper_actions, left_gripper_actions], axis=1)
                gripper_array = np.expand_dims(gripper_array, axis=-1)  # shape: demo_len, 2, 1

                demo_len = gripper_array.shape[0]

                sg_info = f['sg_info']
                sg_params_json = f['sg_params'][()]
                sg_params = json.loads(sg_params_json.decode('utf-8'))
                robot_names = sg_params['robots']   

                ## decide the boundary
                biop_skill_info = sg_info['bimanual_0'] ## TODO: use kw to represent the biop skill
                biop_pre_sg = self._get_sg(biop_skill_info, 'pre_sg')
                biop_eff_sg = self._get_sg(biop_skill_info, 'eff_sg')

                biop_start_idx = biop_pre_sg.graph['idx_list'][0]
                if biop_eff_sg is not None:
                    biop_end_idx = biop_eff_sg.graph['idx_list'][-1]
                else:
                    biop_end_idx = demo_len

                # self.ep_metadata[ep_key] = {
                #     'biop_start_idx': biop_start_idx,
                #     'biop_end_idx': biop_end_idx,
                #     'robot_names': robot_names,
                #     'demo_len': demo_len,
                # }

                ## NOTE: currently train whole episodes. The solution maybe: robot0, robot1 is reversed!
                self.ep_metadata[ep_key] = {
                    'biop_start_idx': 0,
                    'biop_end_idx': demo_len-1,
                    'robot_names': robot_names,
                    'demo_len': demo_len,
                }

                # self.ep_length_dict[ep_key] = biop_end_idx
                

                # ######## transformation size: demo_len， 2， 4， 4
                if self.dof == 10:
                    eef_action_trans = f[f'data/demo_{demo_id}/datagen_info/target_pose'][()]   
                    eef_action_3vec = eef_action_trans[:, :, :3, [3, 0, 1]].transpose(0, 1, 3, 2)
                    eef_action_9d = eef_action_3vec.reshape(demo_len, self.num_eef, 9)
                    eef_action_10d = np.concatenate([gripper_array, eef_action_9d], axis=-1)
  
                    # cache eef actions
                    self.cache[ep_key]['eef10d:action'] = eef_action_10d.reshape(demo_len, -1)
                    # self.cache[ep_key]['gripper:action'] = gripper_array

                ## for original 
                elif self.dof == 7:
                    ## NOTE: in action_dict, robot1 --> left, robot0 --> right
                    left_eef_action_relpos = f[f'data/demo_{demo_id}/action_dict/left_rel_pos'][()]
                    left_eef_action_relrot = f[f'data/demo_{demo_id}/action_dict/left_rel_rot_axis_angle'][()]
                    right_eef_action_relpos = f[f'data/demo_{demo_id}/action_dict/right_rel_pos'][()]
                    right_eef_action_relrot = f[f'data/demo_{demo_id}/action_dict/right_rel_rot_axis_angle'][()]
                    eef_action_relpos = np.concatenate([np.expand_dims(right_eef_action_relpos, axis=1), np.expand_dims(left_eef_action_relpos, axis=1)], axis=1)
                    eef_action_relrot = np.concatenate([np.expand_dims(right_eef_action_relrot, axis=1), np.expand_dims(left_eef_action_relrot, axis=1)], axis=1)
                    eef_action_7d = np.concatenate([gripper_array, eef_action_relpos, eef_action_relrot], axis=-1)
                    self.cache[ep_key]['eef7d:action'] = eef_action_7d.reshape(demo_len, -1)

                ####### get pc, demo_len, N, 3
                obj_pcds =  {}
                for obj_pc_key in f['data/obj_pcd'].keys():
                    raw_vlen = f[f'data/obj_pcd/{obj_pc_key}'][()]
                    pc_list = [raw_vlen[i].reshape(-1, 3) for i in range(len(raw_vlen))]
                    obj_name = obj_pc_key.split('_points')[0]
                    obj_pcds[obj_name] = pc_list


                related_objs = biop_skill_info['related_objs'][()]
                pc_keys = [obj.decode('utf-8') for obj in related_objs]
                related_objs_pc = []
                for i in range(demo_len):
                    all_pc = []
                    for key in pc_keys:
                        all_pc.append(obj_pcds[key][i])
                    related_objs_pc.append(np.concatenate(all_pc, axis=0))

                ## cache pc 
                self.cache[ep_key]['related_objs:pc'] = related_objs_pc

                #######  cache eef states
                obs_grp = f[f'data/demo_{demo_id}/obs']
                eef_states = {}
                gripper_states = {}
                for robot_name in robot_names:
                    eef_states[robot_name] = []
                    biop_eef_pos = obs_grp[f'{robot_name}_eef_pos']
                    biop_eef_quat = obs_grp[f'{robot_name}_eef_quat']
                    gripper_state_2finger = obs_grp[f'{robot_name}_gripper_qpos'][()]
                    gripper_states[robot_name] = gripper_state_2finger[:, 0].reshape(demo_len, 1, 1)
                    for eef_pos, eef_quat in zip(biop_eef_pos, biop_eef_quat):
                        eef_state = compose_transformation(eef_pos, eef_quat)
                        eef_states[robot_name].append(eef_state.reshape(1, 4, 4))

                if self.num_eef == 1:
                    eef_state_trans = eef_states[robot_names[0]]
                    gripper_vals = gripper_states[robot_names[0]]
                else:
                    eef_state_trans = np.concatenate([eef_states[robot_name] for robot_name in robot_names], axis=1)
                    gripper_vals = np.concatenate([gripper_states[robot_name] for robot_name in robot_names], axis=1)

                ########## transformation size: demo_len, 2, 4, 4
                eef_state_3vec = eef_state_trans[:, :, :3, [3, 0, 1]].transpose(0, 1, 3, 2)
                eef_state_9d = eef_state_3vec.reshape(demo_len, self.num_eef, 9)
                gravity_vec = np.array([0, 0, -1])
                gravity_expanded = np.tile(gravity_vec, (demo_len, self.num_eef, 1))
                eef_state_13d = np.concatenate([eef_state_9d, gravity_expanded, gripper_vals], axis=-1)
                # eef_state_10d = np.concatenate([gripper_vals, eef_state_9d], axis=-1)
            
                ## cache states
                self.cache[ep_key]['eef13d:state'] = eef_state_13d
                # self.cache[ep_key]['gripper:state'] = gripper_vals

                for start_id in range(biop_start_idx, biop_end_idx- self.pred_horizon + 1):
                    self.timestep_indices.append((file_idx, start_id))


            pass

                        



    def __getitem__(self, idx):
        file_idx, ep_t = self.timestep_indices[idx]
        file_name = self.hdf5_files[file_idx]
        # demo_id = file_name.split('_')[-2]
        ep_key = f'{file_name}'
        
        # # Calculate temporal window
        # start_t = ep_t - (self.obs_horizon - 1)
        # end_t = ep_t + self.pred_horizon
        # ep_t_list = np.arange(start_t, end_t)
        # clipped_ep_t_list = np.clip(ep_t_list, 0, self.ep_length_dict[ep_key] - 1)
        
        # Handle augmentation
        if self.num_augment > 0:
            if self.same_aug_per_sample:
                aug_idx = np.random.randint(self.num_augment)
            else:
                aug_idx = idx * self.num_augment + np.random.randint(self.num_augment)
        else:
            aug_idx = None

            
        ret = self._get_data_from_cache(ep_key, ep_t, aug_idx)
        
        # Validate shapes
        assert len(ret["pc"]) == self.obs_horizon, f"pc shape: {len(ret['pc'])}, obs_horizon: {self.obs_horizon}"
        assert len(ret["eef_pos"]) == self.obs_horizon, f"eef_pos shape: {len(ret['eef_pos'])}, obs_horizon: {self.obs_horizon}"
        assert len(ret["action"]) == self.pred_horizon, f"action shape: {len(ret['action'])}, pred_horizon: {self.pred_horizon}"
        
        # Reduce horizon dimension if needed
        if self.obs_horizon == 1 and self.pred_horizon == 1 and self.reduce_horizon_dim:
            ret = {k: v[0] for k, v in ret.items()}
            
        return ret



    # def _get_rbt_actions(self, robot_names, obs_grp):
    #     """Extract action for given timestep"""
    #     data_dict = {}
    #     for robot_name in robot_names:
    #         data_dict[f'{robot_name}_joint_pos'] = obs_grp[f'{robot_name}_joint_pos'][()]
    #         data_dict[f'{robot_name}_eef_pos'] = obs_grp[f'{robot_name}_eef_pos'][()]
    #         data_dict[f'{robot_name}_eef_quat'] = obs_grp[f'{robot_name}_eef_quat'][()]

    #     return data_dict

    def _get_sg(self, hdf5_group, sg_name):
        sg_json = hdf5_group[sg_name][()] if sg_name in hdf5_group else None
        if sg_json is None:
            return None
        sg_str = sg_json.decode('utf-8')
        sg = nx.node_link_graph(json.loads(sg_str))
        return sg

    def _downsample_pc(self, xyz):
        choice = np.random.choice(
            xyz.shape[0],
            self.num_points,
            replace=False if xyz.shape[0] >= self.num_points else True,
        )
        if self.mode == "train" and self.shuffle_pc:
            xyz = xyz[choice, :]
        else:
            step = xyz.shape[0] // self.num_points
            xyz = xyz[::step, :][: self.num_points, :] ## uniform sampling
        return xyz 

    def _proc_pc_list(self, pc_list):
        pc_list = [self._downsample_pc(pc) for pc in pc_list]
        return np.array(pc_list)

    def _get_data_from_cache(self, ep_key, ep_t,  aug_idx=None):
        """Get processed data from cache"""
        if ep_key not in self.cache:
            raise KeyError(f"Data not found for {ep_key}")
     
        # Calculate temporal window, obsvation from start_t to ep_t, action from ep_t to end_t
        start_t =  ep_t - self.obs_horizon # ep_t - (self.obs_horizon - 1) Do we have to separate obs and pred?
        end_t = ep_t + self.pred_horizon

        obs_data_keys = [k for k in self.cache[ep_key].keys() if k.endswith(':pc')]
        action_data_keys = [k for k in self.cache[ep_key].keys() if k.endswith(':action')]
        state_data_keys = [k for k in self.cache[ep_key].keys() if k.endswith(':state')]
        # Extract requested keys
        data = {}
        for key in obs_data_keys:
            pc_xyz = self.cache[ep_key][key][start_t:ep_t]
            data["pc"] = self._proc_pc_list(pc_xyz)
        for key in state_data_keys:
            data["eef_pos"] = self.cache[ep_key][key][start_t:ep_t]
        for key in action_data_keys:
            data["action"] = self.cache[ep_key][key][ep_t:end_t]
        
        # Apply data processing (similar to original _process_data_from_file)

        # if "eef_pos" in data:
        #     eef_pos = data["eef_pos"]
        #     eef_pos = eef_pos.reshape(self.num_eef, -1)
        #     eef_pos = eef_pos[:, : self.eef_dim]
        #     data["eef_pos"] = eef_pos

        # if "action" in data:
        #     action = data["action"]
        #     data["action"] = action

        # Apply augmentation
        if self.num_augment > 0 and aug_idx is not None:
            data = self._apply_augmentation(data, aug_idx, data.keys())
            
        # Ensure proper data types
        for key in data:
            data[key] = data[key].astype(np.float32)
                
        return data

    def _apply_augmentation(self, data, aug_idx, keys):
        """Apply data augmentation - copied from original implementation"""
        if aug_idx == 0 and self.aug_keep_original:
            return data
            
        rs = np.random.RandomState(aug_idx)
        
        # Generate augmentation parameters
        if self.aug_scale_aspect_limit > 1.0:
            while True:
                scale = (
                    rs.rand(3) * (self.aug_scale_high - self.aug_scale_low)
                    + self.aug_scale_low
                )
                if scale.max() / scale.min() < 1.33:
                    break
        else:
            scale = np.full(
                (3,),
                rs.rand() * (self.aug_scale_high - self.aug_scale_low)
                + self.aug_scale_low,
            )

        if self.aug_scale_rot < 0:
            rot = rs.rand() * np.pi * 2
        else:
            rot = (rs.rand() * 2 - 1) * self.aug_scale_rot
        offset = rs.randn(3) * self.aug_scale_pos

        if self.aug_zero_z_offset:
            offset[2] = 0

        center = self.aug_center
        
        # Apply augmentation to point cloud
        if "pc" in keys and "pc" in data:
            xyz = data["pc"]
            xyz = rotate_around_z(xyz, rot, center, scale).astype(np.float32)
            xyz += offset[None]
            data["pc"] = xyz
            
        # Apply augmentation to end-effector positions
        if "eef_pos" in keys and "eef_pos" in data:
            eef_pos = data["eef_pos"]
            eef_pos_shape = eef_pos.shape
            if self.dof < 7:
                eef_pos = eef_pos.reshape(-1, 3)
                eef_pos = rotate_around_z(eef_pos, rot, center, scale)
                eef_pos += offset[None]
                eef_pos = eef_pos.reshape(eef_pos_shape).astype(np.float32)
            else:
                assert self.eef_dim in [13, 16]
                eef_pos[:, 0:3] = (
                    rotate_around_z(eef_pos[:, 0:3], rot, center, scale)
                    + offset[None]
                )
                eef_pos[:, 3:6] = rotate_around_z(eef_pos[:, 3:6], rot)
                eef_pos[:, 6:9] = rotate_around_z(eef_pos[:, 6:9], rot)
                if self.eef_dim == 16:
                    eef_pos[:, 13:16] = (
                        rotate_around_z(eef_pos[:, 13:16], rot, center, scale)
                        + offset[None]
                    )
            data["eef_pos"] = eef_pos
            
        # Apply augmentation to actions
        if "action" in keys and "action" in data:
            action = data["action"]
            if self.dof == 3:
                action = action.reshape(-1, 3)
                action = rotate_around_z(action, rot, center, scale)
            elif self.dof == 4:
                action = action.reshape(-1, 4)
                action[:, 1:] = rotate_around_z(
                    action[:, 1:], rot, center, scale
                )
            elif self.dof == 7:
                action = action.reshape(-1, 7)
                action[:, 1:4] = rotate_around_z(
                    action[:, 1:4], rot, center, scale
                )
                action[:, 4:7] = rotate_around_z(action[:, 4:7], rot)
            else:
                raise ValueError(
                    f"Unexpected action shape {action.shape} and dof {self.dof}"
                )
            data["action"] = action
            
        return data
    
@hydra.main(config_path=os.path.join(EQUIBOT_PATH, "equibot/policies/configs"), config_name="dmg_assembly_policy")
def main(cfg):
    import sys
    sys.path.append('/home/user/yzchen_ws/TAMP-ubuntu22/pddlstream_aloha')
    # sys.path.append('/mnt/TAMP/interbotix_ws/src/pddlstream_aloha')
    # sys.path.append('/home/xuhang/interbotix_ws/src/pddlstream_aloha')
    # from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose

    test_dataset = RobosuitePolicyDataset(cfg.data.dataset, "test")
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
        pass


if __name__ == '__main__':
    main()