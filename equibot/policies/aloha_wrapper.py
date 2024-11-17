import os
import sys
import torch
import hydra
import numpy as np

from equibot.policies.utils.misc import get_agent, get_dataset, ActionSlice, to_torch, rotate_observation, to_tensor
from equibot.policies.agents.aloha_agent import ALOHAAgent  
from equibot.policies.agents.compaloha_agent import CompALOHAAgent  

# from equibot.policies.datasets.abstract_dataset import ALOHAPoseDataset
from equibot.policies.datasets.dual_abs_dataset import DualAbsDataset

TAMP_PATH = '/home/xuhang/interbotix_ws/src/pddlstream_aloha/'



class pddl_wrapper(object):
    def __init__(self, cfg, dataset_path):
         # load the network
        cfg.data.dataset.path = dataset_path

        self.cfg = cfg
        self.agent = get_agent(cfg.agent.agent_name)(cfg)
        self.agent.train(False)
        self.agent.load_snapshot(cfg.training.ckpt)

        self.dataset = get_dataset(cfg, cfg.mode)
        # self.dataset = DualAbsDataset(cfg.data.dataset , cfg.mode)

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
    def centralize_obs(self, obs):
        centralized_obs = obs.copy()
        offset_dict = {}
        for k, v in obs.items():
            if 'pc' in k:
                pc = v.numpy().reshape(-1, 3)
                centered_pc, offset = self.dataset.centralize_cond_pc(pc)
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
                decentralized_pc = self.dataset.decentralize_cond_pc(pc, offset_dict[grasp_key])
                # decentralize_obs[k] = decentralized_pc
                decentralize_obs[k] = torch.tensor(decentralized_pc, device= self.cfg.device).reshape(1, 1, -1, 3).float()
        return decentralize_obs
    
    def decentralize_history(self, history, offset_dict, **kwargs):
        for action_slice in history:
            for k, v in offset_dict.items():
                action_slice.data[k] = self.dataset.decentralize_grasp(action_slice.data[k], offset_dict[k], **kwargs)
        return history
    
    def decentralize_action(self, action_dict, offset_dict):
        for k, v in offset_dict.items():
            action_dict[k] = self.dataset.decentralize_grasp(action_dict[k], offset_dict[k])
        return action_dict
    
    def dict_tensor_to_numpy(self, dict_tensor):
        dict_numpy = {}
        for k, v in dict_tensor.items():
            if isinstance(v, torch.Tensor):
                dict_numpy[k] = v.cpu().detach().numpy()
            else:
                dict_numpy[k] = v
        return dict_numpy
    
    def infer_real(self, obs):
        obs_tensor = {}
        for k, v in obs.items():
            obs_tensor[k] = torch.tensor(v).float()
        obs_c, offset_dict = self.centralize_obs(obs_tensor)
        action_dict = self.predict_action(obs_c, offset_dict)
        return action_dict
    
    
    def predict_action(self, obs_c,   offset_dict = None, history_bid = -1, **kwargs):
        # for k, v in obs_c.items():
        #     if v is None:
        #         continue
        #     obs_c[k] = torch.tensor(v).float()

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


            vis_sides = None
            if self.cfg.data.dataset.dataset_type =='mj_insertion_pred' :
                vis_sides = ['left', 'right']

            render_history(history_w, use_gui=True, \
                        directory = history_pic_dir, save_pic_every = -1,
                        agent_obs = self.decentralize_obs(obs_gpu, offset_dict),
                        has_eff = self.dataset.has_eff, vis_sides = vis_sides)
            
        if offset_dict is not None:
            action_w = self.decentralize_action(action_c, offset_dict)
        else:
            action_w = action_c
        return action_w



def infer_and_render(dataset_path, config_name, overrides, ply_paths = None, history_bid = -1, **kwargs):
    with hydra.initialize(config_path="configs", job_name="test_app"):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    
    assert cfg.mode != "train"

    np.random.seed(cfg.seed)

    tamp_wrapper = pddl_wrapper(cfg, dataset_path)

    ## if pc is in the world frame. No need to normalize it and get the offset, as center will be calculated in actor
    if ply_paths is  None:
        agent_obs = tamp_wrapper.get_obs_from_datset(**kwargs)
        obs_c = to_torch(agent_obs, tamp_wrapper.cfg.device)
        offset_dict = None
    else:
        agent_obs = tamp_wrapper.get_obs_from_ply(ply_paths, **kwargs)    
        obs_c, offset_dict = tamp_wrapper.centralize_obs(agent_obs)

    action_dict = tamp_wrapper.predict_action(history_bid=history_bid, obs_c=obs_c, offset_dict=offset_dict)



    return action_dict



def main():
    # # mj sim
    # dataset_path = '/home/xuhang/Desktop/yzchen_ws/equibot_abstract/data/mj_peg_hole/'
    # config_name = "mj_peg_hole"
    # overrides = ["prefix=mj_peg_hole", "mode=eval", "use_wandb=false"]
    # ply_paths = {'left_pc': os.path.join(dataset_path, 'left_pc.ply'), 'right_pc': os.path.join(dataset_path, 'right_pc.ply')}
    # # ply_paths = None

    ## aloha transfer tape
    import pathlib
    dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute()
    config_name = "transfer_tape"
    overrides = ["prefix=aloha_transfer_tape", "mode=inference", "use_wandb=false"]
    ply_paths = {'pc': os.path.join(dataset_path, 'tape_OOD.ply')}

    action_dict = infer_and_render(dataset_path, config_name, overrides, ply_paths=ply_paths, history_bid=0)
    print(action_dict)


def eval_with_rotation(ply_name = 'tape_OOD.ply', history_bid = -1):


    ## aloha transfer tape
    import pathlib
    dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute()
    config_name = "transfer_tape"
    overrides = ["prefix=aloha_transfer_tape", "mode=inference", "use_wandb=false"]
    ply_paths = {'pc': os.path.join(dataset_path, ply_name)}
    

    with hydra.initialize(config_path="configs", job_name="test_app"):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    
    assert cfg.mode != "train"

    np.random.seed(cfg.seed)

    tamp_wrapper = pddl_wrapper(cfg, dataset_path)

    agent_obs = tamp_wrapper.get_obs_from_ply(ply_paths)    
    obs_c, offset_dict = tamp_wrapper.centralize_obs(agent_obs)

    raw_action_dict = tamp_wrapper.predict_action(obs_c=obs_c, offset_dict=offset_dict, history_bid=history_bid)
    ref_grasp_angle = raw_action_dict['grasp'][:3, :3]


    rot_to_apply_ls = [np.pi/2, np.pi, 3*np.pi/2]
    for rot in rot_to_apply_ls:
        from equibot.envs.sim_mobile.utils.transformations import euler2mat
        rot_3x3 = euler2mat([0, 0, rot])
        rotated_ref_grasp = np.dot(rot_3x3, ref_grasp_angle)

        agent_obs = tamp_wrapper.get_obs_from_ply(ply_paths, yaw_rotation=rot)    
        obs_c, offset_dict = tamp_wrapper.centralize_obs(agent_obs)
        input_obs = rotate_observation(agent_obs, rot)
        action_dict = tamp_wrapper.predict_action(obs_c=input_obs,    
                                                   offset_dict=offset_dict,
                                                   history_bid=history_bid,
                                                #    ref_grasp=rotated_ref_grasp,
                                                   )
        
        
        pred_grasp_angle = action_dict['grasp'][:3, :3]
        



        print('rotation diff: ', rotation_diff(pred_grasp_angle, rotated_ref_grasp))
        


def rotation_diff(rot1, rot2):
    theta = np.arccos(np.clip((np.trace(np.dot(rot1, rot2.T)) - 1) / 2, -1.0, 1.0))
    deg = np.rad2deg(theta)
    return deg









if __name__ == "__main__":
    # main()
    eval_with_rotation(ply_name='mug_ID.ply', history_bid=0)
