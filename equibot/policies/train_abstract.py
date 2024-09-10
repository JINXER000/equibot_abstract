import os
import sys
import copy
import hydra
import torch
import wandb
import omegaconf
import numpy as np
import getpass as gt
from tqdm import tqdm
from glob import glob
from omegaconf import OmegaConf

from equibot.policies.utils.media import save_video
from equibot.policies.utils.misc import get_env_class, get_dataset, get_agent
from equibot.policies.vec_eval import run_eval
from equibot.envs.subproc_vec_env import SubprocVecEnv

from equibot.policies.datasets.abstract_dataset import ALOHAPoseDataset
from equibot.policies.agents.aloha_agent import ALOHAAgent  

from test_abstract import run_eval
# from torch.utils.tensorboard import SummaryWriter

@hydra.main(config_path="/home/xuhang/Desktop/yzchen_ws/equibot_abstract/equibot/policies/configs", config_name="transfer_tape")
def main(cfg):
    assert cfg.mode == "train"
    np.random.seed(cfg.seed)

    # initialize parameters
    batch_size = cfg.training.batch_size

    # setup logging
    # # tensorboard
    # writer = SummaryWriter()

    # wandb
    if cfg.use_wandb:
        wandb_config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False
        )
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            tags=["train"],
            name=cfg.prefix,
            settings=wandb.Settings(code_dir="."),
            config=wandb_config,
        )
    log_dir = os.getcwd()
    train_dataset = ALOHAPoseDataset(cfg.data.dataset, "train")
    num_workers = cfg.data.dataset.num_workers
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    
    cfg.data.dataset.num_training_steps = (
        cfg.training.num_epochs * len(train_dataset) // batch_size
    )

        # init agent
    agent = ALOHAAgent(cfg)
    if cfg.training.ckpt is not None:
        agent.load_snapshot(cfg.training.ckpt)
        start_epoch_ix = int(cfg.training.ckpt.split("/")[-1].split(".")[0][4:])
    else:
        start_epoch_ix = 0



    # train loop
    min_eval_rot_error = 1e9
    global_step = 0
    for epoch_ix in tqdm(range(start_epoch_ix, cfg.training.num_epochs)):
        batch_ix = 0
        for batch in tqdm(train_loader, leave=False, desc="Batches"):
            train_metrics = agent.update(
                batch, vis=epoch_ix % cfg.training.vis_interval == 0 and batch_ix == 0
            )
            if cfg.use_wandb:
                wandb.log(
                    {"train/" + k: v for k, v in train_metrics.items()},
                    step=global_step,
                )
                wandb.log({"train/epoch": epoch_ix}, step=global_step)

            # # # to tensorboard
            # for k, v in train_metrics.items():
            #     v_ts_cpu = torch.tensor(v)
            #     agent.actor.writer.add_scalar(f"train/{k}", v_ts_cpu, epoch_ix)
            # agent.actor.writer.flush()
            
            del train_metrics
            global_step += 1
            batch_ix += 1



        # run eval 
        if (
            (
                epoch_ix % cfg.training.eval_interval == 0
                or epoch_ix == cfg.training.num_epochs - 1
            )
            and epoch_ix > 0
        ):
            eval_metrics = run_eval(agent = agent, vis= False, batch= batch, history_bid= -1 )
            if cfg.use_wandb:
                wandb.log(
                    {"eval/" + k: v for k, v in eval_metrics.items()},
                    step=global_step,
                )
            rot_error = eval_metrics["grasp_rotation_error"]
            if rot_error < min_eval_rot_error:
                min_eval_rot_error = rot_error
                agent.save_snapshot(os.path.join(log_dir, "ckpt_best.pth"))

            # # # to tensorboard
            # train_step = epoch_ix * len(train_loader)+batch_ix
            # for k, v in eval_metrics.items():
            #     v_ts_cpu = torch.tensor(v)
            #     agent.actor.writer.add_scalar(f"train/{k}", v_ts_cpu, train_step)
            # agent.actor.writer.flush()

        # save ckpt
        if (
            epoch_ix % cfg.training.save_interval == 0
            or epoch_ix == cfg.training.num_epochs - 1
        ):
            save_path = os.path.join(log_dir, f"ckpt{epoch_ix:05d}.pth")
            num_ckpt_to_keep = 10
            if len(list(glob(os.path.join(log_dir, "ckpt*.pth")))) > num_ckpt_to_keep:
                # remove old checkpoints
                for fn in list(sorted(glob(os.path.join(log_dir, "ckpt*.pth"))))[
                    :-num_ckpt_to_keep
                ]:
                    os.remove(fn)
            agent.save_snapshot(save_path)

    # agent.actor.writer.close()


if __name__ == "__main__":
    main()
