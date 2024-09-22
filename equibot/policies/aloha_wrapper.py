import os
import sys
import torch
import hydra
import numpy as np

from equibot.policies.agents.aloha_agent import ALOHAAgent  
from equibot.policies.datasets.abstract_dataset import ALOHAPoseDataset



# from torch.utils.tensorboard import SummaryWriter


# required when input raw point cloud Nx3
def preprocess_pc(points, tgt_size):
    input_pc = np.asarray(points)
    assert input_pc.shape[1] == 3
    pc_min = np.min(input_pc, axis=0)
    input_pc = input_pc - pc_min
    sampled_indices = np.random.choice(input_pc.shape[0], tgt_size, replace=False)
    input_pc = input_pc[sampled_indices]
    return input_pc, pc_min

def postprocess_xyz(trans, pc_min):
    trans[:3, 3] += pc_min
    trans[4:7, 3] += pc_min
    return trans



class pddl_wrapper(object):
    def __init__(self, cfg):
         # load the network
        self.agent = ALOHAAgent(cfg)
        self.agent.train(False)
        self.agent.load_snapshot(cfg.training.ckpt)
        self.has_eff = (cfg.data.dataset.dataset_type == 'hdf5_predeff')
        self.pc_min = np.zeros(3)




    def predict(self, points, history_bid = -1, require_preprocess = True):
        points_vis = points.copy()
        if require_preprocess:
            points, pc_min = preprocess_pc(points, self.agent.cfg.data.dataset.num_points)

        points_batch = torch.tensor(points).float().cuda().reshape(1, 1, -1, 3)
        agent_obs = {"pc": points_batch, "gt_grasp": None, 'joint_pose': None}
        history, action_dict = self.agent.actor(agent_obs, history_bid=history_bid)

        if history_bid >=0:
            log_dir = os.getcwd()
            # log_dir = '/home/xuhang/Desktop/yzchen_ws/equibot_abstract/logs/eval/aloha_transfer_tape'
            history_pic_dir = os.path.join(log_dir, "history_pics")
            if not os.path.exists(history_pic_dir):
                os.makedirs(history_pic_dir)

            # points_np = points_batch[history_bid,0].cpu().numpy()
                
            if require_preprocess:
                history = [(postprocess_xyz(action_slice[0], pc_min), action_slice[1]) for action_slice in history]

            sys.path.append('/home/xuhang/interbotix_ws/src/pddlstream_aloha/')
            from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose

            render_pose(history, use_gui=True, \
                        directory = history_pic_dir, save_pic_every = 10,
                        obj_points = points_vis,
                        has_eff = self.has_eff)
            
        if require_preprocess:
            action_dict['grasp'] = postprocess_xyz(action_dict['grasp'], pc_min)
        return action_dict






def isolated_main():
    # https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="configs", job_name="test_app"):
        cfg = compose(config_name="transfer_tape", overrides=["prefix=aloha_transfer_tape", "mode=eval", "use_wandb=false"])
    

    # print(OmegaConf.to_yaml(cfg))

    assert cfg.mode == "eval"

    np.random.seed(cfg.seed)

# # ######### use dataset as test input
# #     # get eval datase
# #     cfg.data.dataset.path='/home/xuhang/Desktop/yzchen_ws/equibot_abstract/data/transfer_tape/'
# #     eval_dataset = ALOHAPoseDataset(cfg.data.dataset, "test")
# #     num_workers = cfg.data.dataset.num_workers
# #     test_loader = torch.utils.data.DataLoader(
# #         eval_dataset,
# #         batch_size=1,
# #         num_workers=num_workers,
# #         shuffle=True,
# #         drop_last=True,
# #         pin_memory=True,
# #     )

# #     data_iter = iter(test_loader)
# #     fist_batch = next(data_iter)
# #     input_pc = fist_batch['pc'][0].cpu().numpy()
# #     require_preprocess = False

######### use o3d point cloud as test input
    import open3d as o3d
    pcd = o3d.io.read_point_cloud("leaky.ply")
    input_pc = np.asarray(pcd.points)
    require_preprocess = True

    tamp_wrapper = pddl_wrapper(cfg)
    action_dict = tamp_wrapper.predict(input_pc, history_bid = 0, require_preprocess=require_preprocess)

    print(action_dict)

if __name__ == "__main__":
    isolated_main()
