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

sys.path.append('/home/xuhang/interbotix_ws/src/pddlstream_aloha/')
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose
import open3d as o3d


def rotate_points(conditional_pc, visualize=False, rot_z = None):
    points = np.asarray(conditional_pc)

    theta = rot_z
    # # rotate the pc around y axis for 90 deg, then rotate around x axis for 45 deg
    # rotation_y = np.array([
    #     [np.cos(np.pi / 2), 0, np.sin(np.pi / 2)],
    #     [0, 1, 0],
    #     [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)]
    # ])

    # # 45 degrees rotation around the x-axis
    # rotation_x = np.array([
    #     [1, 0, 0],
    #     [0, np.cos(np.pi / 4), -np.sin(np.pi / 4)],
    #     [0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    # ])

    rotation_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Apply the rotations
    # points_rotated = points @ rotation_y.T @ rotation_x.T
    points_rotated = points @ rotation_z.T

    # apply translation
    # points_rotated += np.array([-0.1, -0.1, 0.2])

    if visualize:
        # visualize the pc with open3d
        conditional_pcd = o3d.geometry.PointCloud()
        conditional_pcd.points = o3d.utility.Vector3dVector(points_rotated[0,0])

        # draw axis
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([conditional_pcd, axis])

    return points_rotated   

def ply2points(ply_path, rot_z = None, **kwargs):

    conditional_pc = o3d.io.read_point_cloud(ply_path)
    if rot_z is not None:
        points = rotate_points(conditional_pc.points, rot_z = rot_z)
    else:
        points = np.asarray(conditional_pc.points)

    return points

def process_batch(batch, agent):

    pc = batch["pc"].cpu().numpy()
    grasp_pose = batch["grasp_pose"].cpu().numpy()
    joint_pose = batch["joint_pose"].cpu().numpy()

    # # perform transformation
    # pc = rotate_points(pc)
    
    return pc, grasp_pose, joint_pose

def run_eval(
    agent,
    vis=False,
    log_dir=None,
    use_wandb=False,
    batch = None,
    history_bid = 0,
    **kwargs
):
    

    ## input obs from dataset
    if batch is not None:
        points_batch, gt_grasp_9d = process_batch(batch, agent)
        agent_obs = {"pc": points_batch, "gt_grasp": gt_grasp_9d}
    else:
        # # input dummy obs
        ply_path = "/home/xuhang/Desktop/yzchen_ws/equibot_abstract/data/transfer_tape/tape.ply"
        points = ply2points(ply_path, **kwargs)
        points_batch = points.reshape(1, 1, -1, 3)  # batch size, Ho, N, 3
        agent_obs = {"pc": points_batch}
    


    # predict actions
    st = time.time()
    unnormed_history, metrics = agent.act(agent_obs, history_bid=history_bid)
    print(f": {time.time() - st:.3f}s")

    if vis and history_bid >=0:
        history_pic_dir = os.path.join(log_dir, "history_pics")
        if not os.path.exists(history_pic_dir):
            os.makedirs(history_pic_dir)
        render_pose(unnormed_history, use_gui=True, \
                    directory = history_pic_dir, save_pic_every = 10,
                    obj_points = points_batch[history_bid,0])


    return metrics


@hydra.main(config_path="configs", config_name="fold_synthetic")
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
    cfg.data.dataset.path='/home/xuhang/Desktop/yzchen_ws/equibot_abstract/data/transfer_tape/'
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

        rotate_yaw_list = [0, np.pi/2, np.pi, np.pi/2*3]

        for rot_z in rotate_yaw_list:
            eval_metrics = run_eval(
                agent,
                vis=True,
                log_dir=log_dir,
                batch = None, # fist_batch,
                history_bid = cfg.eval.history_bid,
                rotate_yaw_list = rotate_yaw_list,
                rot_z = rot_z,
            )
        # print metrics
        print(f"ckpt: {ckpt_name}, eval_metrics: {eval_metrics}")
    #     for k, v in eval_metrics.items():
    #         writer.add_scalar(f"eval/{k}", v, i)

    # writer.flush()
    # writer.close()


if __name__ == "__main__":
    main()
