import os
import sys
import time
import torch
import hydra
import omegaconf
import wandb
import numpy as np
import getpass as gt
from glob import glob
from tqdm import tqdm

from equibot.policies.utils.media import combine_videos, save_video
from equibot.policies.agents.aloha_agent import ALOHAAgent  
from equibot.policies.datasets.abstract_dataset import ALOHAPoseDataset

sys.path.append('/home/user/yzchen_ws/TAMP-ubuntu22/pddlstream_aloha')
sys.path.append('/mnt/TAMP/interbotix_ws/src/pddlstream_aloha')
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose
import open3d as o3d

# from torch.utils.tensorboard import SummaryWriter

def rotate_points(conditional_pc, visualize=False):
    points = np.asarray(conditional_pc)

    # rotate the pc around y axis for 90 deg, then rotate around x axis for 45 deg
    rotation_y = np.array([
        [np.cos(np.pi / 2), 0, np.sin(np.pi / 2)],
        [0, 1, 0],
        [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)]
    ])

    # 45 degrees rotation around the x-axis
    rotation_x = np.array([
        [1, 0, 0],
        [0, np.cos(np.pi / 4), -np.sin(np.pi / 4)],
        [0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    ])

    # Apply the rotations
    points_rotated = points @ rotation_y.T @ rotation_x.T
    # points_rotated = points

    # apply translation
    points_rotated += np.array([-0.1, -0.1, 0.2])

    if visualize:
        # visualize the pc with open3d
        conditional_pcd = o3d.geometry.PointCloud()
        conditional_pcd.points = o3d.utility.Vector3dVector(points_rotated[0,0])

        # draw axis
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([conditional_pcd, axis])

    return points_rotated   

def ply2points(ply_path):

    conditional_pcd = o3d.io.read_point_cloud(ply_path)
    # points = rotate_points(conditional_pcd.points)
    points = np.asarray(conditional_pcd.points)

    return points

def process_batch(batch, agent):

    pc = batch["pc"].cpu().numpy()
    grasp_pose = batch["grasp_pose"].cpu().numpy()
    joint_pose = batch["joint_pose"].cpu().numpy()

    # # perform transformation
    # pc = rotate_points(pc)

    # ###### Below is to test if input random point cloud, the prediction of jpose is ok
    # points_mean = np.mean(pc, axis=-2, keepdims=True)
    # points_var = np.var(pc, axis=-2, keepdims=True)
    # random_points = np.random.randn(pc.shape[0], pc.shape[1], pc.shape[2], pc.shape[3])
    # random_points = random_points - np.mean(random_points, axis=-2, keepdims=True) + points_mean
    # random_points = random_points * np.sqrt(points_var) / np.sqrt(np.var(random_points, axis=-2, keepdims=True))
    # pc = random_points

    return pc, grasp_pose, joint_pose



def run_eval(
    agent,
    vis=False,
    log_dir=None,
    use_wandb=False,
    batch = None,
    history_bid = 0,
    has_eff = False
):
    

    ## input obs from dataset
    if batch is not None:
        points_batch, gt_grasp_9d, joint_pose = process_batch(batch, agent)
    else:
        # # input dummy obs
        ply_path = "/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/data/transfer_tape/raw/graspobj_4.ply"
        points = ply2points(ply_path)
        points_batch = points.reshape(1, 1, -1, 3)  # batch size, Ho, N, 3

    agent_obs = {"pc": points_batch, "gt_grasp": gt_grasp_9d, 'joint_pose': joint_pose}


    # predict actions
    st = time.time()
    unnormed_history, metrics = agent.act(agent_obs, history_bid=history_bid)
    print(f"Inference time: {time.time() - st:.3f}s")

    if vis and history_bid >=0:
        history_pic_dir = os.path.join(log_dir, "history_pics")
        if not os.path.exists(history_pic_dir):
            os.makedirs(history_pic_dir)
        render_pose(unnormed_history, use_gui=True, \
                    directory = history_pic_dir, save_pic_every = 10,
                    obj_points = points_batch[history_bid,0],
                    has_eff = has_eff)


    return metrics


@hydra.main(config_path="configs", config_name="transfer_tape")
def main(cfg):
    assert cfg.mode == "eval"
    device = torch.device(cfg.device)
    if cfg.use_wandb:
        wandb_config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False
        )
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            tags=["eval"],
            name=cfg.prefix,
            settings=wandb.Settings(code_dir="."),
            config=wandb_config,
        )
    np.random.seed(cfg.seed)


    # get eval datase
    cfg.data.dataset.path='/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/data/transfer_tape/'
    eval_dataset = ALOHAPoseDataset(cfg.data.dataset, "test")
    num_workers = cfg.data.dataset.num_workers
    test_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=32,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    data_iter = iter(test_loader)
    fist_batch = next(data_iter)

    agent = ALOHAAgent(cfg)
    agent.train(False)

    # draw encoder in tensorboard
    # writer = SummaryWriter()
    # dummy_input = torch.randn(32, 2, 512, 3, device=device).float()
    # writer.add_graph(agent.actor.encoder, dummy_input)


    if os.path.isdir(cfg.training.ckpt):
        ckpt_dir = cfg.training.ckpt
        ckpt_paths = list(glob(os.path.join(ckpt_dir, "ckpt*.pth")))
        assert len(ckpt_paths) >= cfg.eval.num_ckpts_to_eval
        ckpt_paths = list(sorted(ckpt_paths))[-cfg.eval.num_ckpts_to_eval :]
        assert f"{cfg.eval.last_ckpt}" in ckpt_paths[-1]
    else:
        ckpt_paths = [cfg.training.ckpt]

    for i, ckpt_path in enumerate(ckpt_paths):
        ckpt_name = ckpt_path.split("/")[-1].split(".")[0]
        agent.load_snapshot(ckpt_path)

        log_dir = os.getcwd()

        eval_metrics = run_eval(
            agent,
            vis=True,
            log_dir=log_dir,
            batch = fist_batch,
            history_bid = cfg.eval.history_bid,
            has_eff = (cfg.data.dataset.dataset_type == 'hdf5_predeff')
        )
    #     # print metrics
    #     print(f"ckpt: {ckpt_name}, eval_metrics: {eval_metrics}")
    #     for k, v in eval_metrics.items():
    #         writer.add_scalar(f"eval/{k}", v, i)

    # writer.flush()
    # writer.close()


if __name__ == "__main__":
    main()
