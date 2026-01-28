import os
import sys
import torch
import hydra
import numpy as np

from equibot.policies.utils.misc import get_agent, get_agent_from_ckpt, get_dataset, to_np, to_torch, rotate_observation, to_tensor, EQUIBOT_PATH, decentralize_cond_pc, decentralize_grasp, centralize_downsample, combined_pc_instances_and_offset


TAMP_PATH = '/home/xuhang/interbotix_ws/src/pddlstream_aloha/'



class pddl_wrapper(object):


    def __init__(self, dataset_path, cfg = None, ckpt_path = None, exe_mode = "inference"):
         # load the network
        if cfg is not None and ckpt_path is None:
            ckpt_path_full = os.path.join(EQUIBOT_PATH, cfg.training.ckpt)
            self.agent = get_agent(cfg.agent.agent_name)(cfg)
            self.agent.train(False)
            self.agent.load_snapshot(ckpt_path_full)
        elif ckpt_path is not None and cfg is None:
            ckpt_path_full = os.path.join(EQUIBOT_PATH, ckpt_path)
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
                pc = v.numpy().reshape(-1, 3)
                centered_pc, offset = centralize_downsample(pc, self.dataset.pc_shape, add_bottom = is_add_bottom, **kwargs)
                centralized_obs[k] = torch.tensor(centered_pc, device= self.cfg.device).reshape(1, 1, -1, 3).float()
                
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

            sys.path.append(TAMP_PATH)
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

    def gen_objcentric_traj(self, obs_key, agent_obs, skill_name = None, task_name = None):

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
        action_c, eval_metrics = self.agent.actor.pred_unimaual_traj(skill_key, obs_gpu, task_name_batch=task_name)
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
    
    def gen_bimanual_kp(self,  related_pc_dict, skill_name = None, task_name = None):
        obs_tensor = to_tensor(related_pc_dict)
        # obs_c, offset_dict = self.centralize_obs(obs_tensor, obj_centric=self.cfg.data.dataset.is_obj_centric, method=self.cfg.data.dataset.downsample_method)
        init_pc_n, init_pc_offset= combined_pc_instances_and_offset(obs_tensor, self.dataset.pc_shape, self.dataset.is_obj_centric, self.dataset.is_add_bottom, self.dataset.downsample_method)
        obs_c = {'pc': init_pc_n}
        offset_dict = {'eefpos': init_pc_offset}
        obs_c = to_tensor(obs_c)
        obs_gpu = to_torch(obs_c, self.cfg.device)
        
        skill_key = skill_name
        action_c, eval_metrics = self.agent.actor.pred_unimaual_traj(skill_key, obs_gpu, task_name_batch=task_name)
        action_c = to_np(action_c)

        ## TODO: decode for bimanual

        ## decentralize the final action
        action_w = self.decentralize_action(action_c, offset_dict)


        return action_w
    

    def gen_uncond_jposes(self, arm1, arm2, sk):
        ## old version
        # action_dict, eval__metrics = self.agent.actor.pred_bimanual_jposes(sk, batch_size = 1)
        # jpose_out = to_np(action_dict)[f'{sk}:jpose']

        ## per_skill version
        action_dict, eval__metrics = self.agent.actor.pred_bimanual_jposes(sk, agent_obs = None)
        jpose_out = to_np(action_dict)['jpose']

        return jpose_out


def get_obsc_offset_dict(tamp_wrapper, ply_paths = None, obj_centric = False, **kwargs):
    ## if pc is in the world frame. No need to normalize it and get the offset, as center will be calculated in actor
    if ply_paths is  None:
        agent_obs = tamp_wrapper.get_obs_from_datset(**kwargs)
        obs_c = to_torch(agent_obs, tamp_wrapper.cfg.device)
        offset_dict = None
    else:
        agent_obs = tamp_wrapper.get_obs_from_ply(ply_paths, **kwargs)    
        obs_c, offset_dict = tamp_wrapper.centralize_obs(agent_obs, obj_centric = obj_centric)
    return obs_c, offset_dict

def get_cfgs(task_name):
    if task_name == 'mj_peg_hole':
        # mj sim
        dataset_path = 'data/mj_peg_hole/'
        config_name = "mj_peg_hole"
        overrides = ["prefix=mj_peg_hole", "mode=inference", "use_wandb=false"]
        # ply_paths = {'left_pc': os.path.join(dataset_path, 'left_pc.ply'), 'right_pc': os.path.join(dataset_path, 'right_pc.ply')}
        ply_paths = None
    elif task_name == 'aloha_transfer_tape':
        ## aloha transfer tape
        import pathlib
        dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute()
        config_name = "transfer_tape"
        overrides = ["prefix=aloha_transfer_tape", "mode=inference", "use_wandb=false"]
        ply_paths = {'pc': os.path.join(dataset_path, 'tape.ply')}
    elif task_name == 'aloha_handoff_cup':
        ## aloha transfer tape
        import pathlib
        dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute()
        config_name = "handoff_cup"
        overrides = ["prefix=aloha_handoff_cup", "mode=inference", "use_wandb=false"]
        ply_paths = {'pc': os.path.join(dataset_path, 'singleview_cup.ply')}
    elif 'screwdriver' in task_name:
        ## screwdriver and its variants
        import pathlib
        dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute()
        config_name = task_name
        overrides = ["prefix="+task_name, "mode=inference", "use_wandb=false"]
        ply_paths = {'pc': os.path.join(dataset_path, 'debug_screwdriver.ply')}    
    else:
        raise NotImplementedError('task not implemented')
    return dataset_path, config_name, overrides, ply_paths


def rot_mat_from_action_dict(action_dict):
    rot_dict = {}
    for key in action_dict.keys():
        if 'grasp' in key:
            rot_dict[key] = action_dict[key][:3, :3]
    return rot_dict

def eval_with_rotation(task_name = 'screwdriver', history_bid = -1):
    dataset_path, config_name, overrides, ply_paths = get_cfgs(task_name)
    
    with hydra.initialize(config_path="configs", job_name="test_app"):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    
    # assert cfg.mode != "train"
    cfg.mode = 'eval'
    # cfg.mode = 'inference'
    np.random.seed(cfg.seed)

    tamp_wrapper = pddl_wrapper(cfg, dataset_path)

    ## if pc is in the world frame. No need to normalize it and get the offset, as center will be calculated in actor
    if ply_paths is  None:
        agent_obs = tamp_wrapper.get_obs_from_datset()
        obs_c = to_torch(agent_obs, tamp_wrapper.cfg.device)
        offset_dict = None
    else:
        agent_obs = tamp_wrapper.get_obs_from_ply(ply_paths)    ## tensor cpu
        obs_c, offset_dict = tamp_wrapper.centralize_obs(agent_obs, obj_centric = cfg.data.dataset.is_obj_centric)  ## tensor gpu

    # obs_c, offset_dict = get_obsc_offset_dict(tamp_wrapper, ply_paths, obj_centric = cfg.data.dataset.is_obj_centric)

    raw_action_dict = tamp_wrapper.predict_action(obs_c=obs_c, offset_dict=offset_dict, history_bid=history_bid, sleep_time=0.05)
    # ref_grasp_dict = rot_mat_from_action_dict(raw_action_dict)


    rot_to_apply_ls = np.arange(np.pi/3, 2*np.pi, np.pi/3)
    # rot_to_apply_ls = [2*np.pi]
    for rot in rot_to_apply_ls:

        rotated_obs_np = rotate_observation(agent_obs, rot)
        ## infer_real = centralize + predict_action + decentralize
        action_dict = tamp_wrapper.infer_real(rotated_obs_np, history_bid=history_bid)
       
        # pred_grasp_angle = rot_mat_from_action_dict(action_dict)  

        # rot_diff = rot_diff_from_dicts(ref_grasp_dict, pred_grasp_angle, gt_rot_euler=rot)
        
def rot_diff_from_dicts(ori_dict, pred_dict, gt_rot_euler):
    from equibot.envs.sim_mobile.utils.transformations import euler2mat
    rot_3x3 = euler2mat([0, 0, gt_rot_euler])
    rot_diff = {}
    for key in ori_dict.keys():
        rotated_ref_grasp = np.dot(rot_3x3, ori_dict[key]) 
        rot_diff[key] = rotation_diff(rotated_ref_grasp, pred_dict[key])
        print('rotation diff for {} is {}'.format(key, rot_diff[key]))
    return rot_diff

def rotation_diff(rot1, rot2):
    # theta = np.arccos(np.clip((np.trace(np.dot(rot1, rot2.T)) - 1) / 2, -1.0, 1.0))
    # deg = np.rad2deg(theta)
    from scipy.spatial.transform import Rotation as R
    q1 = R.from_matrix(rot1).as_quat()
    q2 = R.from_matrix(rot2).as_quat()
    q_relative = R.from_quat(q2) * R.from_quat(q1).inv()
    euler_angles = q_relative.as_euler('xyz', degrees=True)

    return euler_angles


if __name__ == "__main__":
    # main()
    # eval_with_rotation(task_name='screwdriver_container', history_bid=0)
    eval_with_rotation(task_name='mj_peg_hole', history_bid=0)
