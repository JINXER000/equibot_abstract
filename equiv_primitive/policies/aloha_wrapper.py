import os
import sys
import torch
import hydra
import numpy as np
sys.path.append('.')
from equiv_primitive.policies.utils.misc import get_agent, get_agent_from_ckpt, get_dataset, to_np, to_torch, rotate_observation, to_tensor, EQUIV_PRIMITIVE_PATH, decentralize_cond_pc, decentralize_grasp, centralize_downsample, combined_pc_instances_and_offset, geodestDist, add_pcd_noise
from equiv_primitive.policies.utils.repo_paths import env_path


class pddl_wrapper(object):


    def __init__(self, dataset_path, cfg = None, ckpt_path = None, exe_mode = "inference"):
         # load the network
        if cfg is not None and ckpt_path is None:
            ckpt_path_full = os.path.join(EQUIV_PRIMITIVE_PATH, cfg.training.ckpt)
            self.agent = get_agent(cfg.agent.agent_name)(cfg)
            self.agent.train(False)
            self.agent.load_snapshot(ckpt_path_full)
        elif ckpt_path is not None and cfg is None:
            ckpt_path_full = os.path.join(EQUIV_PRIMITIVE_PATH, ckpt_path)
            self.agent = get_agent_from_ckpt(ckpt_path_full)
            cfg = self.agent.cfg

        cfg.mode = exe_mode
        cfg.data.dataset.path = dataset_path
        self.cfg = cfg
        self.agent.train(False)

        self.dataset = get_dataset(cfg, cfg.mode)

        if cfg.mode != 'inference':
            # num_workers = cfg.data.dataset.num_workers
            num_workers = 0
            self.test_loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                num_workers=num_workers,
                shuffle=False,
                drop_last=True,
                pin_memory=True,
            )

    def get_skill_names(self):
        return list(self.agent.actor.skill_names)

    def get_obs_from_datset(self,**kwargs):
        assert self.cfg.mode != 'inference'
        data_iter = iter(self.test_loader)
        fist_batch = next(data_iter)

        return fist_batch
    
    def get_obs_from_ply(self, ply_paths = {}, **kwargs):
        assert self.cfg.mode == 'inference'
        import open3d as o3d
        data_batch = {}
        for k, v in ply_paths.items():
            pcd = o3d.io.read_point_cloud(v)
            input_pc = np.asarray(pcd.points)
            
            data_batch[k] = torch.tensor(input_pc).unsqueeze(0).unsqueeze(0).float()
        return data_batch

    ## do not use it during training
    def centralize_obs(self, obs, is_add_bottom = False, **kwargs):
        centralized_obs = obs.copy()
        offset_dict = {}
        for k, v in obs.items():
            if 'pc' in k:
                pc = v.numpy()
                centered_pc, offset = centralize_downsample(pc, self.dataset.pc_shape, add_bottom = is_add_bottom, **kwargs)
                centered_pc= torch.tensor(centered_pc, device= self.cfg.device).float()
                centralized_obs[k] = centered_pc.unsqueeze(0).unsqueeze(0)
                
                grasp_key = k.replace('pc', 'grasp')

                offset_dict[grasp_key] = offset

        return centralized_obs, offset_dict
    
    def decentralize_obs(self, obs, offset_dict = None):
        if offset_dict is None:
            return obs
        decentralize_obs = obs.copy()
        for k, v in obs.items():
            if 'pc' in k:
                v_cpu = v.cpu().detach()
                pc = v_cpu.numpy().reshape(-1, 3)
                grasp_key = k.replace('pc', 'grasp')
                decentralized_pc = decentralize_cond_pc(pc, offset_dict[grasp_key])
                # decentralize_obs[k] = decentralized_pc
                decentralize_obs[k] = torch.tensor(decentralized_pc, device= self.cfg.device).reshape(1, 1, -1, 3).float()
        return decentralize_obs
    
    def decentralize_history(self, history, offset_dict, **kwargs):
        for action_slice in history:
            for k, v in offset_dict.items():
                action_slice.data[k] = decentralize_grasp(action_slice.data[k], offset_dict[k], **kwargs)
        return history
    
    def decentralize_action(self, action_dict_c, offset_dict):
        action_dict = action_dict_c.copy()
        for k, v in offset_dict.items():
            if k not in action_dict.keys():
                continue
            action_dict[k] = decentralize_grasp(action_dict[k], offset_dict[k])
        return action_dict
    
    def dict_tensor_to_numpy(self, dict_tensor):
        dict_numpy = {}
        for k, v in dict_tensor.items():
            if isinstance(v, torch.Tensor):
                dict_numpy[k] = v.cpu().detach().numpy()
            else:
                dict_numpy[k] = v
        return dict_numpy
    
    def infer_real(self, obs, is_add_bottom = False, **kwargs):
        obs_tensor = to_tensor(obs)
        obs_c, offset_dict = self.centralize_obs(obs_tensor, obj_centric=self.cfg.data.dataset.is_obj_centric, is_add_bottom = is_add_bottom)
        action_dict = self.predict_action(obs_c, offset_dict, **kwargs)
        return action_dict
    
    
    def predict_action(self, obs_c,   offset_dict = None, history_bid = -1, **kwargs):

        obs_c = to_tensor(obs_c)

        obs_gpu = to_torch(obs_c, self.cfg.device)
        
        action_c, eval_metrics, history_c = \
            self.agent.actor(obs_gpu, history_bid=history_bid)
        action_c = self.dict_tensor_to_numpy(action_c)

        if history_bid >=0:
            log_dir = os.getcwd()
            history_pic_dir = os.path.join(log_dir, "history_pics")
            if not os.path.exists(history_pic_dir):
                os.makedirs(history_pic_dir)

            if offset_dict is not None:
                ## move the gripper to the world frame
                history_w =  self.decentralize_history(history_c, offset_dict, **kwargs)
            else:
                history_w = history_c

            sys.path.append(env_path("TAMP_ROOT"))
            from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose, render_history


            render_history(history_w, use_gui=True, \
                        directory = history_pic_dir, save_pic_every = -1,
                        agent_obs = self.decentralize_obs(obs_gpu, offset_dict),
                        vis_eff = False, #self.dataset.has_eff, 
                        **kwargs)
        ## decentralize the final action
        if offset_dict is not None:
            action_w = self.decentralize_action(action_c, offset_dict)
        else:
            action_w = action_c
        return action_w

    def predict_skill_keyposes_world(self, object_pc, skill_name, task_name, seed=None, pcd_noise=0.0):
        """Predict a unimanual per-skill eef keypose trajectory in the world frame.

        This is the object-centric inference path used for sim-observation eval:
        the raw point cloud is centered the same way the dataset preprocesses it,
        the per-skill diffusion policy predicts an object-centric eef trajectory,
        and the result is shifted back to the world frame by the centering offset.

        Args:
            object_pc: (N, 3) point cloud of the conditioning object, world frame.
            skill_name: checkpoint skill key, e.g. ``robot0_grasp_tripod_obj``.
            task_name: task embedding key, e.g. ``two_arm_threading``.
            seed: optional diffusion seed (None -> stochastic sample).
            pcd_noise: std (meters) of Gaussian xyz jitter added to the centered
                point cloud, matching the training-time ``add_pcd_noise``
                augmentation (0 -> no jitter).

        Returns:
            dict with ``eefpos`` (T, 4, 4) world-frame SE(3) keyposes and
            ``gripper`` (T, ...) as numpy arrays.
        """
        ds_cfg = self.cfg.data.dataset
        centered_pc, offset = centralize_downsample(
            np.asarray(object_pc, dtype=np.float32)[:, :3],
            self.dataset.pc_shape,
            obj_centric=ds_cfg.is_obj_centric,
            add_bottom=ds_cfg.is_add_bottom,
            method=ds_cfg.downsample_method,
            debug_visualize=False,
        )
        if pcd_noise > 0:
            centered_pc = add_pcd_noise(centered_pc, pcd_noise)
        pc_tensor = torch.tensor(
            centered_pc, device=self.cfg.device, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        self.agent.actor.train(False)
        with torch.no_grad():
            action_c, _ = self.agent.actor.pred_unimanual_traj(
                skill_name, {"pc": pc_tensor}, task_name_batch=task_name, seed=seed
            )
        action_c = self.dict_tensor_to_numpy(action_c)
        eefpos_world = decentralize_grasp(action_c["eefpos"].copy(), offset)
        return {"eefpos": eefpos_world, "gripper": action_c["gripper"]}

    def keypose_error(self, pred_T_world, gt_T_world):
        """Position (m) and rotation (deg, geodesic) error between two SE(3) trajectories.

        Args:
            pred_T_world: (T, 4, 4) predicted world-frame keyposes.
            gt_T_world: (T, 4, 4) ground-truth world-frame keyposes.

        Returns:
            dict with ``pos_err_m`` and ``rot_err_deg`` (means over the T keyposes).
        """
        pred_T = np.asarray(pred_T_world, dtype=np.float32)
        gt_T = np.asarray(gt_T_world, dtype=np.float32)
        assert pred_T.shape == gt_T.shape, f"shape mismatch {pred_T.shape} vs {gt_T.shape}"

        pos_err = np.linalg.norm(pred_T[:, :3, 3] - gt_T[:, :3, 3], axis=-1).mean()
        rot_rad = geodestDist(
            torch.from_numpy(gt_T[:, :3, :3]), torch.from_numpy(pred_T[:, :3, :3])
        ).mean()
        return {
            "pos_err_m": float(pos_err),
            "rot_err_deg": float(torch.rad2deg(rot_rad)),
        }

    def gen_objcentric_traj(
        self, obs_key, agent_obs, skill_name=None, task_name=None, seed=None
    ):

        if seed is not None:
            np.random.seed(seed)
        import re
        def revise_key(action_output, key_mapping):
            new_action_output = {}
            for key, value in action_output.items():
                # First do the regex replacements
                new_key = re.sub(r'robot0[^:]*:', 'left_', key)
                new_key = re.sub(r'robot1[^:]*:', 'right_', new_key)
                # Then do the specific key mappings
                for origin_k, revised_k in key_mapping:
                    new_key = new_key.replace(origin_k, revised_k)
                new_action_output[new_key] = value
            return new_action_output
        
        obs_tensor = to_tensor(agent_obs)
        obs_c, offset_dict = self.centralize_obs(obs_tensor, obj_centric=self.cfg.data.dataset.is_obj_centric, method=self.cfg.data.dataset.downsample_method, is_add_bottom = True)
        obs_c = to_tensor(obs_c)
        obs_gpu = to_torch(obs_c, self.cfg.device)
        
        skill_key = skill_name if skill_name is not None else obs_key 

        if task_name is None:
            raise ValueError('task_name is required for non-libero tasks')
        
        self.agent.actor.train(False)
        with torch.no_grad():
            action_c, eval_metrics = self.agent.actor.pred_unimanual_traj(
                skill_key, obs_gpu, task_name_batch=task_name, seed=seed
            )
        action_c = to_np(action_c)

        # key_mapping = [('robot0_grasp_piece_1:','left_'),('eefpos','grasp'),\
                    #    ('robot1_grasp_piece_2:','right_'), ('robot0_piece_1_contact_base:','left_')]
        key_mapping = [('eefpos','grasp')]
        action_c = revise_key(action_c, key_mapping)
        offset_dict = revise_key(offset_dict, key_mapping)
        ## decentralize the final action
        if offset_dict is not None:
            action_w = self.decentralize_action(action_c, offset_dict)
        else:
            action_w = action_c

        return action_w
    
    def gen_bimanual_kp(
        self, related_pc_dict, skill_name=None, task_name=None, seed=None
    ):
        obs_tensor = to_tensor(related_pc_dict)
        # obs_c, offset_dict = self.centralize_obs(obs_tensor, obj_centric=self.cfg.data.dataset.is_obj_centric, method=self.cfg.data.dataset.downsample_method)
        init_pc_n, init_pc_offset= combined_pc_instances_and_offset(obs_tensor, self.dataset.pc_shape, self.dataset.is_obj_centric, self.dataset.is_add_bottom, self.dataset.downsample_method)
        obs_c = {'pc': init_pc_n}
        offset_dict = {'eefpos': init_pc_offset}
        obs_c = to_tensor(obs_c)
        obs_gpu = to_torch(obs_c, self.cfg.device)
        
        skill_key = skill_name
        self.agent.actor.train(False)
        with torch.no_grad():
            action_c, eval_metrics = self.agent.actor.pred_unimanual_traj(
                skill_key, obs_gpu, task_name_batch=task_name, seed=seed
            )
        action_c = to_np(action_c)

        ## TODO: decode for bimanual

        ## decentralize the final action
        action_w = self.decentralize_action(action_c, offset_dict)


        return action_w
    

    def gen_uncond_jposes(self, arm1, arm2, sk, seed=None):
        ## old version
        # action_dict, eval__metrics = self.agent.actor.pred_bimanual_jposes(sk, batch_size = 1)
        # jpose_out = to_np(action_dict)[f'{sk}:jpose']

        ## per_skill version
        action_dict, eval__metrics = self.agent.actor.pred_bimanual_jposes(
            sk, agent_obs=None, seed=seed
        )
        jpose_out = to_np(action_dict)['jpose']

        return jpose_out
