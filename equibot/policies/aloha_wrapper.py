import os
import sys
import torch
import hydra
import numpy as np

from equibot.policies.utils.misc import get_agent, get_dataset, ActionSlice, to_torch
from equibot.policies.agents.aloha_agent import ALOHAAgent  
from equibot.policies.agents.compaloha_agent import CompALOHAAgent  

# from equibot.policies.datasets.abstract_dataset import ALOHAPoseDataset
from equibot.policies.datasets.dual_abs_dataset import DualAbsDataset

TAMP_PATH = '/home/xuhang/interbotix_ws/src/pddlstream_aloha/'


def rotate_pc(pc, yaw = 0):
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw), np.cos(yaw), 0],
                  [0, 0, 1]])
    return np.dot(pc, R.T)

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




    def get_obs_from_datset(self, applied_transform = None,**kwargs):
        assert self.cfg.mode != 'inference'
        data_iter = iter(self.test_loader)
        fist_batch = next(data_iter)

        ## TODO: below is not checked!
        if applied_transform is not None:
            for k, v in fist_batch.items():
                if 'pc' in k:
                    fist_batch[k] = torch.tensor(rotate_pc(v.numpy().reshape(-1, 3), applied_transform)).reshape(1, 1, -1, 3).float()
        return fist_batch
    
    def get_obs_from_ply(self, ply_paths = {}, applied_transform = None, **kwargs):
        assert self.cfg.mode == 'inference'
        import open3d as o3d
        data_batch = {}
        for k, v in ply_paths.items():
            pcd = o3d.io.read_point_cloud(v)
            input_pc = np.asarray(pcd.points)
            
            if applied_transform is not None:
                input_pc = rotate_pc(input_pc, applied_transform)
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
    
    def decentralize_history(self, history, offset_dict):
        for action_slice in history:
            for k, v in offset_dict.items():
                action_slice.data[k] = self.dataset.decentralize_grasp(action_slice.data[k], offset_dict[k])
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
    
    def predict_action(self, obs_c,   offset_dict = None, history_bid = -1):
        for k, v in obs_c.items():
            if v is None:
                continue
            obs_c[k] = torch.tensor(v).float()
        
        action_c, eval_metrics, history_c = \
            self.agent.actor(obs_c, history_bid=history_bid)
        action_c = self.dict_tensor_to_numpy(action_c)

        if history_bid >=0:
            log_dir = os.getcwd()
            history_pic_dir = os.path.join(log_dir, "history_pics")
            if not os.path.exists(history_pic_dir):
                os.makedirs(history_pic_dir)

            if offset_dict is not None:
                ## move the gripper to the world frame
                history_w =  self.decentralize_history(history_c, offset_dict)
            else:
                history_w = history_c

            sys.path.append(TAMP_PATH)
            from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose, render_history


            vis_sides = None
            if self.cfg.data.dataset.dataset_type =='mj_insertion_pred' :
                vis_sides = ['left', 'right']

            render_history(history_w, use_gui=True, \
                        directory = history_pic_dir, save_pic_every = -1,
                        agent_obs = self.decentralize_obs(obs_c, offset_dict),
                        has_eff = self.dataset.has_eff, vis_sides = vis_sides)
            
        if offset_dict is not None:
            action_w = self.decentralize_action(action_c, offset_dict)
        else:
            action_w = action_c
        return action_w



def infer_and_render(dataset_path, config_name, overrides, ply_paths = None, **kwargs):
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

    action_dict = tamp_wrapper.predict_action(history_bid=0, obs_c=obs_c, offset_dict=offset_dict)

    ## TODO: if agent_obs has grasp value, then compare the difference.
    if 'grasp' in agent_obs:
        raise NotImplementedError('grasp rotation diff not implemented')
        grasp_diff = agent_obs['grasp'] - action_dict['grasp']
        print('grasp diff: ', grasp_diff)
        action_dict['se3_diff'] = grasp_diff

    return action_dict



def main():
    # # mj sim
    # dataset_path = '/home/chenyizhou/imitation_learning/equibot_abstract/data/mj_peg_hole/'
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

    action_dict = infer_and_render(dataset_path, config_name, overrides, ply_paths=ply_paths)
    print(action_dict)


def eval_with_rotation():


    ## aloha transfer tape
    import pathlib
    dataset_path = pathlib.Path(__file__).parent.parent.parent.absolute()
    config_name = "transfer_tape"
    overrides = ["prefix=aloha_transfer_tape", "mode=inference", "use_wandb=false"]
    ply_paths = {'pc': os.path.join(dataset_path, 'tape_OOD.ply')}

    action_dict = infer_and_render(dataset_path, config_name, overrides, ply_paths=ply_paths, applied_transform = np.pi/2)
    print(action_dict)

    





if __name__ == "__main__":
    main()
