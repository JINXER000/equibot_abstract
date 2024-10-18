import os
import sys
import torch
import hydra
import numpy as np

from equibot.policies.utils.misc import get_agent, get_dataset, ActionSlice, to_torch
# from equibot.policies.agents.aloha_agent import ALOHAAgent  
# from equibot.policies.agents.compaloha_agent import CompALOHAAgent  

# from equibot.policies.datasets.abstract_dataset import ALOHAPoseDataset
from equibot.policies.datasets.dual_abs_dataset import DualAbsDataset



# from torch.utils.tensorboard import SummaryWriter


# # required when input raw point cloud Nx3
# def preprocess_pc(points, tgt_size, is_mj = False):
#     input_pc = np.asarray(points)
#     assert input_pc.shape[1] == 3
#     pc_min = np.min(input_pc, axis=0)
#     input_pc = input_pc - pc_min
#     sampled_indices = np.random.choice(input_pc.shape[0], tgt_size, replace=False)
#     input_pc = input_pc[sampled_indices]
#     return input_pc, pc_min

# def postprocess_xyz(trans, pc_min):
#     trans[:3, 3] += pc_min
#     trans[4:7, 3] += pc_min
#     return trans



class pddl_wrapper(object):
    def __init__(self, cfg):
         # load the network
        self.cfg = cfg
        self.agent = get_agent(cfg.agent.agent_name)(cfg)
        self.agent.train(False)
        self.agent.load_snapshot(cfg.training.ckpt)

        # self.dataset = get_dataset(cfg, "train")
        self.dataset = DualAbsDataset(cfg.data.dataset , "test")
        num_workers = cfg.data.dataset.num_workers
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )

        self.vis_side = None
        if cfg.data.dataset.dataset_type =='mj_insertion_pred' :
            self.vis_side = 'left'


    def get_obs_from_datset(self):
        data_iter = iter(self.test_loader)
        fist_batch = next(data_iter)
        return fist_batch
    
    def get_obs_from_ply(self, ply_paths = {}):
        import open3d as o3d
        for k, v in ply_paths.items():
            pcd = o3d.io.read_point_cloud(v)
            input_pc = np.asarray(pcd.points)
            data_batch = {}
            data_batch[k] = torch.tensor(input_pc).unsqueeze(0).unsqueeze(0).float()
        return data_batch

    ## do not use it during training
    def centralize_obs(self, obs):
        centralize_obs = obs.copy()
        offset_dict = {}
        for k, v in obs.items():
            if 'pc' in k:
                pc = v.numpy().reshape(-1, 3)
                centered_pc, offset = self.dataset.centralize_cond_pc(pc)
                centralize_obs[k] = torch.tensor(centered_pc, device= self.cfg.device).reshape(1, 1, -1, 3).float()
                
                grasp_key = k.replace('pc', 'grasp')
                centralize_obs[grasp_key] = None
                joint_key = k.replace('pc', 'jpose') 
                centralize_obs[joint_key] = None

                offset_dict[grasp_key] = offset

        return centralize_obs, offset_dict
    
    def decentralize_history(self, history, offset_dict):
        for action_slice in history:
            for k, v in offset_dict:
                action_slice.data[k] = self.dataset.decentralize_grasp(action_slice.data[k], offset_dict[k])
        return history
    
    def decentralize_action(self, action_dict, offset_dict):
        for k, v in offset_dict.items():
            action_dict[k] = self.dataset.decentralize_grasp(action_dict[k], offset_dict[k])
        return action_dict
    
    def predict_action(self, history_bid = -1, ply_paths = None):
        if ply_paths is  None:
            agent_obs = self.get_obs_from_datset()
            obs_c = to_torch(agent_obs, self.cfg.device)
        else:
            agent_obs = self.get_obs_from_ply(ply_paths)    
            obs_c, offset_dict = self.centralize_obs(agent_obs)


        ## if pc is in the world frame. No need to normalize it and get the offset, as center will be calculated in actor
        
        action_c, eval_metrics, history_c = \
            self.agent.actor(obs_c, history_bid=history_bid)

        if history_bid >=0:
            log_dir = os.getcwd()
            history_pic_dir = os.path.join(log_dir, "history_pics")
            if not os.path.exists(history_pic_dir):
                os.makedirs(history_pic_dir)

            if ply_paths is not None:
                ## move the gripper to the world frame
                history_w =  self.decentralize_history(history_c, offset_dict)
            else:
                history_w = history_c

            sys.path.append('/home/xuhang/interbotix_ws/src/pddlstream_aloha/')
            from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose, render_history


            render_history(history_w, use_gui=True, \
                        directory = history_pic_dir, save_pic_every = 10,
                        agent_obs = agent_obs,
                        has_eff = self.dataset.has_eff, side = self.vis_side)
            
        if ply_paths is not None:
            action_w = self.decentralize_action(action_c, offset_dict)
        else:
            action_w = action_c
        return action_w



def infer_and_render(dataset_path, config_name, overrides, ply_paths = None):
    with hydra.initialize(config_path="configs", job_name="test_app"):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    
    assert cfg.mode == "eval"

    np.random.seed(cfg.seed)

    cfg.data.dataset.path = dataset_path
    tamp_wrapper = pddl_wrapper(cfg)


    action_dict = tamp_wrapper.predict_action(ply_paths = None, history_bid=0)

    return action_dict

def main():
    dataset_path = '/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/data/mj_peg_hole/'
    config_name = "mj_peg_hole"
    overrides = ["prefix=mj_peg_hole", "mode=eval", "use_wandb=false"]
    action_dict = infer_and_render(dataset_path, config_name, overrides)
    print(action_dict)



if __name__ == "__main__":
    main()
