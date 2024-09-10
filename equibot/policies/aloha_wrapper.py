import os
import sys
import torch
import hydra
import numpy as np


sys.path.append('/home/user/yzchen_ws/TAMP-ubuntu22/pddlstream_aloha')
sys.path.append('/mnt/TAMP/interbotix_ws/src/pddlstream_aloha')
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose
from equibot.policies.agents.aloha_agent import ALOHAAgent  
from equibot.policies.datasets.abstract_dataset import ALOHAPoseDataset

# from torch.utils.tensorboard import SummaryWriter


class pddl_wrapper(object):
    def __init__(self, cfg):
         # load the network
        self.agent = ALOHAAgent(cfg)
        self.agent.train(False)
        self.agent.load_snapshot(cfg.training.ckpt)
        self.has_eff = has_eff = (cfg.data.dataset.dataset_type == 'hdf5_predeff')


    def predict(self, points, history_bid = -1):
        points_batch = torch.tensor(points).float().cuda().reshape(1, 1, -1, 3)
        agent_obs = {"pc": points_batch, "gt_grasp": None, 'joint_pose': None}
        history, action_dict = self.agent.actor(agent_obs, history_bid=history_bid)

        if history_bid >=0:
            # log_dir = os.getcwd()
            log_dir = '/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/logs/eval/aloha_transfer_tape'
            history_pic_dir = os.path.join(log_dir, "history_pics")
            if not os.path.exists(history_pic_dir):
                os.makedirs(history_pic_dir)

            points_np = points_batch[history_bid,0].cpu().numpy()
            render_pose(history, use_gui=True, \
                        directory = history_pic_dir, save_pic_every = 10,
                        obj_points = points_np,
                        has_eff = self.has_eff)
            
        
        return action_dict


# @hydra.main(config_path="configs", config_name="transfer_tape")
# def main(cfg):
#     assert cfg.mode == "eval"

#     np.random.seed(cfg.seed)

#     # get eval datase
#     cfg.data.dataset.path='/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/data/transfer_tape/'
#     eval_dataset = ALOHAPoseDataset(cfg.data.dataset, "test")
#     num_workers = cfg.data.dataset.num_workers
#     test_loader = torch.utils.data.DataLoader(
#         eval_dataset,
#         batch_size=1,
#         num_workers=num_workers,
#         shuffle=True,
#         drop_last=True,
#         pin_memory=True,
#     )

#     data_iter = iter(test_loader)
#     fist_batch = next(data_iter)

#     tamp_wrapper = pddl_wrapper(cfg)
#     action_dict = tamp_wrapper.predict(fist_batch['pc'][0].cpu().numpy(), history_bid = 0)

#     print(action_dict)





def isolated_main():
    # https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="configs", job_name="test_app"):
        cfg = compose(config_name="transfer_tape", overrides=["prefix=aloha_transfer_tape", "mode=eval", "use_wandb=false"])
    

    # print(OmegaConf.to_yaml(cfg))

    assert cfg.mode == "eval"

    np.random.seed(cfg.seed)

# ######### use dataset as test input
#     # get eval datase
#     cfg.data.dataset.path='/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/data/transfer_tape/'
#     eval_dataset = ALOHAPoseDataset(cfg.data.dataset, "test")
#     num_workers = cfg.data.dataset.num_workers
#     test_loader = torch.utils.data.DataLoader(
#         eval_dataset,
#         batch_size=1,
#         num_workers=num_workers,
#         shuffle=True,
#         drop_last=True,
#         pin_memory=True,
#     )

#     data_iter = iter(test_loader)
#     fist_batch = next(data_iter)
#     input_pc = fist_batch['pc'][0].cpu().numpy()

######### use o3d point cloud as test input
    import open3d as o3d
    pcd = o3d.io.read_point_cloud("debugdiffgen.ply")
    input_pc = np.asarray(pcd.points)


    tamp_wrapper = pddl_wrapper(cfg)
    action_dict = tamp_wrapper.predict(input_pc, history_bid = 0)

    print(action_dict)

if __name__ == "__main__":
    isolated_main()
