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

from equibot.policies.agents.compaloha_agent import CompALOHAAgent  
from equibot.policies.datasets.dual_abs_dataset import DualAbsDataset

sys.path.append('/home/user/yzchen_ws/TAMP-ubuntu22/pddlstream_aloha')
sys.path.append('/mnt/TAMP/interbotix_ws/src/pddlstream_aloha')
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose



def get_obs(batch):

    right_pc = batch["right_pc"].cpu().numpy()
    right_grasp = batch["right_grasp"].cpu().numpy()
    left_jpose = batch["left_jpose"].cpu().numpy()
    right_jpose = batch["right_jpose"].cpu().numpy()

    # # perform transformation
    # pc = rotate_points(pc)
    
    agent_obs = {"right_pc": right_pc, "right_grasp": right_grasp, 'left_jpose': left_jpose, 'right_jpose': right_jpose}
    return agent_obs


def run_eval(
    agent,
    vis=False,
    log_dir=None,
    use_wandb=False,
    batch = None,
    history_bid = 0,
):

    # ## input obs from dataset
    # agent_obs = get_obs(batch)

    # predict actions
    st = time.time()
    unnormed_history, metrics = agent.act(batch, history_bid=history_bid)
    # print(f"Inference time: {time.time() - st:.3f}s")

    if vis and history_bid >=0:
        history_pic_dir = os.path.join(log_dir, "history_pics")
        if not os.path.exists(history_pic_dir):
            os.makedirs(history_pic_dir)

        points_batch = batch['right_pc']
        render_pose(unnormed_history, use_gui=True, \
                    directory = history_pic_dir, save_pic_every = 10,
                    obj_points = points_batch[history_bid,0])


    return unnormed_history, metrics


@hydra.main(config_path="configs", config_name="dual_transfer_tape")
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
    cfg.data.dataset.path='/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/data/mj_peg_hole/'
    eval_dataset = DualAbsDataset(cfg.data.dataset, "test")
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

    agent = CompALOHAAgent(cfg)
    agent.train(False)




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

        unnormed_history, eval_metrics = run_eval(
            agent,
            vis=True,
            log_dir=log_dir,
            batch = fist_batch,
            history_bid = cfg.eval.history_bid,
        )




if __name__ == "__main__":
    main()
