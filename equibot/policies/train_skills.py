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

from equibot.policies.utils.misc import EQUIBOT_PATH, get_agent, get_dataset

from .test_skills import run_eval

@hydra.main(config_path=os.path.join(EQUIBOT_PATH, "equibot/policies/configs"), config_name="transfer_tape")
def main(cfg):
    assert cfg.mode == "train"
    np.random.seed(cfg.seed)

    # initialize parameters
    batch_size = cfg.training.batch_size

    train_dataset = get_dataset(cfg, "train")
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
    agent = get_agent(cfg.agent.agent_name)(cfg)
    if cfg.training.ckpt is not None:
        agent.load_snapshot(cfg.training.ckpt)
        start_epoch_ix = int(cfg.training.ckpt.split("/")[-1].split(".")[0][4:])
    else:
        start_epoch_ix = 0

    # wandb
    if cfg.use_wandb:
        log_dir = os.getcwd()
        cur_date = os.popen("date +'%Y-%m-%d_%H-%M-%S'").read().strip()
        log_dir = os.path.join(log_dir, f"{cur_date}", 'checkpoints')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
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
    else:
        log_dir = None

    # train loop
    # min_eval_rot_error = 1e9
    global_step = 0
    for epoch_ix in tqdm(range(start_epoch_ix, cfg.training.num_epochs)):
        batch_ix = 0
        for batch in tqdm(train_loader, leave=False, desc="Batches"):
            train_metrics = agent.update(batch)
            if cfg.use_wandb:
                wandb.log(
                    {"train/" + k: v for k, v in train_metrics.items()},
                    step=global_step,
                )
                wandb.log({"train/epoch": epoch_ix}, step=global_step)

            
            del train_metrics
            global_step += 1
            batch_ix += 1



        # run eval 
        if ( # log_dir is not None and
            (
                epoch_ix % cfg.training.eval_interval == 0
                or epoch_ix == cfg.training.num_epochs - 1
            )
            and epoch_ix > 0
        ):
            _, eval_metrics = run_eval(agent = agent, vis= False, batch= batch, history_bid= -1 )
            if cfg.use_wandb:
                wandb.log(
                    {"eval/" + k: v for k, v in eval_metrics.items()},
                    step=global_step,
                )

            # agent.save_snapshot(os.path.join(log_dir, "ckpt_best.pth"))


        # save ckpt
        if log_dir is not None and (
            epoch_ix % cfg.training.save_interval == 0
            or epoch_ix == cfg.training.num_epochs - 1
        ):
            save_path = os.path.join(log_dir, f"ckpt{epoch_ix:05d}.pth")
            num_ckpt_to_keep = 1
            if len(list(glob(os.path.join(log_dir, "ckpt*.pth")))) > num_ckpt_to_keep:
                # remove old checkpoints
                for fn in list(sorted(glob(os.path.join(log_dir, "ckpt*.pth"))))[
                    :-num_ckpt_to_keep
                ]:
                    os.remove(fn)
            agent.save_snapshot(save_path)


if __name__ == "__main__":
    main()
