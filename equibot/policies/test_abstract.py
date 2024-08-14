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
from equibot.policies.utils.misc import to_torch

from equibot.policies.utils.media import combine_videos, save_video
from equibot.policies.agents.aloha_agent import ALOHAAgent  
from equibot.policies.datasets.abstract_dataset import ALOHAPoseDataset

sys.path.append('/home/user/yzchen_ws/TAMP-ubuntu22/pddlstream_aloha')
sys.path.append('/mnt/TAMP/interbotix_ws/src/pddlstream_aloha')
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose
import open3d as o3d

def rotate_points(conditional_pc):
    points = np.asarray(conditional_pc.points)

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

    # apply translation
    points_rotated += np.array([0.1, 2.1, 0.1])
    # Update the point cloud with the rotated points
    conditional_pc.points = o3d.utility.Vector3dVector(points_rotated)

    # draw axis
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([conditional_pc, axis])

    return points_rotated   

def ply2points(ply_path):

    conditional_pc = o3d.io.read_point_cloud(ply_path)
    # points = rotate_points(conditional_pc)
    points = np.asarray(conditional_pc.points)

    return points

def process_batch(batch, agent):
    pc = batch["pc"].reshape(-1, 3).cpu().numpy()
    grasp_pose = batch["grasp_pose"].reshape(-1).cpu().numpy()
    return pc, grasp_pose

def run_eval(
    agent,
    vis=False,
    log_dir=None,
    use_wandb=False,
    first_batch = None
):
    if vis:
        vis_frames = []

        ## input obs from dataset
        if first_batch is not None:
            points, gt_grasp_9d = process_batch(first_batch, agent)
        else:
            # # input dummy obs
            ply_path = "/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/data/transfer_tape/raw/graspobj_4.ply"
            points = ply2points(ply_path)

        agent_obs = {"pc": [points], "gt_grasp": [gt_grasp_9d]}


        # predict actions
        st = time.time()
        unnormed_history = agent.act(agent_obs, return_history=True)
        print(f"Inference time: {time.time() - st:.3f}s")

        history_pic_dir = os.path.join(log_dir, "history_pics")
        if not os.path.exists(history_pic_dir):
            os.makedirs(history_pic_dir)
        render_pose(unnormed_history, use_gui=True, \
                    directory = history_pic_dir, save_pic_every = 10,
                    obj_points = points)


        # take actions
        for ac_ix in range(ac_horizon):
            if len(obs["pc"]) == 0 or len(obs["pc"][0]) == 0:
                ac_dict = None
                break
            agent_ac = ac[ac_ix] if len(ac.shape) > 1 else ac
            state, rew, done, info = env.step(agent_ac, dummy_reward=True)
            if hasattr(env, "visualize_eef_frame"):
                env.visualize_eef_frame(state)
            rgb_render = render = env.render()
            obs = organize_obs(render, rgb_render, state)
            obs_history.append(obs)
            if len(obs) > obs_horizon:
                obs_history = obs_history[-obs_horizon:]
            images[-1].append(rgb_render["images"][0][..., :3])
            if vis:
                vis_frames.append(rgb_render["images"][0][..., :3])




    max_num_images = np.max([len(images[i]) for i in range(len(images))])
    for i in range(len(images)):
        if len(images[i]) < max_num_images:
            images[i] = images[i] + [images[i][-1]] * (max_num_images - len(images[i]))
    images = np.array(images)
    rews = np.array(rews)

    pos_idxs, neg_idxs = np.where(rews >= 0.5)[0], np.where(rews < 0.5)[0]
    metrics = dict(rew=np.mean(rews))
    fps = 30 if "sim_mobile" in env.__module__ else 4
    if use_wandb:
        if len(pos_idxs) > 0:
            metrics["video_pos"] = wandb.Video(
                combine_videos(images[pos_idxs][:6], num_cols=5).transpose(0, 3, 1, 2),
                fps=30,
            )
        if len(neg_idxs) > 0:
            metrics["video_neg"] = wandb.Video(
                combine_videos(images[neg_idxs][:6], num_cols=5).transpose(0, 3, 1, 2),
                fps=30,
            )
        if vis:
            metrics["vis_rollout"] = images
            metrics["vis_pc"] = wandb.Object3D(sample_pc)
    else:
        metrics["vis_rollout"] = images
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
    cfg.data.dataset.path='/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/data/transfer_tape/'
    eval_dataset = ALOHAPoseDataset(cfg.data.dataset, "test")
    num_workers = cfg.data.dataset.num_workers
    test_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    data_iter = iter(test_loader)
    first_batch = next(data_iter)

    agent = ALOHAAgent(cfg)
    agent.train(False)

    if os.path.isdir(cfg.training.ckpt):
        ckpt_dir = cfg.training.ckpt
        ckpt_paths = list(glob(os.path.join(ckpt_dir, "ckpt*.pth")))
        assert len(ckpt_paths) >= cfg.eval.num_ckpts_to_eval
        ckpt_paths = list(sorted(ckpt_paths))[-cfg.eval.num_ckpts_to_eval :]
        assert f"{cfg.eval.last_ckpt}" in ckpt_paths[-1]
    else:
        ckpt_paths = [cfg.training.ckpt]

    rew_list = []

    for ckpt_path in ckpt_paths:
        ckpt_name = ckpt_path.split("/")[-1].split(".")[0]
        agent.load_snapshot(ckpt_path)

        log_dir = os.getcwd()

        eval_metrics = run_eval(
            agent,
            vis=True,
            log_dir=log_dir,
            first_batch = first_batch,
        )
        mean_rew = eval_metrics["rew"]
        print(f"Evaluation results: mean rew = {mean_rew}")
        rew_list.append(mean_rew)
        if cfg.use_wandb:
            wandb.log(
                {"eval/" + k: v for k, v in eval_metrics.items() if k != "vis_rollout"}
            )
        else:
            save_filename = os.path.join(
                os.getcwd(), f"vis_{ckpt_name}_rew{mean_rew:.3f}.mp4"
            )
        if "vis_rollout" in eval_metrics:
            if len(eval_metrics["vis_rollout"].shape) == 4:
                save_video(eval_metrics["vis_rollout"], save_filename, fps=30)
            else:
                assert len(eval_metrics["vis_rollout"][0].shape) == 4
                for eval_idx, eval_video in enumerate(eval_metrics["vis_rollout"]):
                    episode_rew = eval_metrics["rew_values"][eval_idx]
                    save_filename = os.path.join(
                        os.getcwd(),
                        f"vis_{ckpt_name}_ep{eval_idx}_rew{episode_rew:.3f}.mp4",
                    )
                    save_video(eval_video, save_filename)
        del eval_metrics
    np.savez(os.path.join(os.getcwd(), "info.npz"), rews=np.array(rew_list))


if __name__ == "__main__":
    main()
