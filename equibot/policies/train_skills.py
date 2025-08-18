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
from torch.utils.data import random_split

from equibot.policies.utils.misc import EQUIBOT_PATH, get_agent, get_dataset

try:
    from .test_skills import run_eval
except ImportError:
    from test_skills import run_eval

@hydra.main(config_path=os.path.join(EQUIBOT_PATH, "equibot/policies/configs"), config_name="transfer_tape")
def main(cfg):
    assert cfg.mode == "train"
    np.random.seed(cfg.seed)

    # initialize parameters
    batch_size = cfg.training.batch_size

    # Load the full dataset
    full_dataset = get_dataset(cfg, "train")
    
    # Split dataset into train and validation
    total_size = len(full_dataset)
    train_size = int(0.98 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed)
    )
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    num_workers = cfg.data.dataset.num_workers
    
    # Import the collate_fn from the dataset module
    from equibot.policies.datasets.per_skill_dataset import collate_fn
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Create validation dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # No shuffling for validation
        drop_last=False,  # Keep all validation samples
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    cfg.data.dataset.num_training_steps = (
        cfg.training.num_epochs * len(train_dataset) // batch_size
    )

        # init agent
    agent = get_agent(cfg.agent.agent_name)(cfg)
    if cfg.training.ckpt is not None:
        agent.load_snapshot(cfg.training.ckpt)
        start_epoch_ix = int(cfg.training.ckpt.split("/")[-1].split(".")[0][4:]) ## format: ckptxxxxx.pth
    else:
        start_epoch_ix = 0

    ## copy the normalizer from dataset
    agent.set_normalizer_and_statistics(full_dataset)

    # wandb
    if cfg.use_wandb:
        log_dir = os.getcwd()
        print(f"log_dir: {log_dir}")
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
        # ## debug
        # log_dir = os.getcwd()

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

        # run eval on validation dataset
        if ( # log_dir is not None and
            (
                epoch_ix % cfg.training.eval_interval == 0
                or epoch_ix == cfg.training.num_epochs - 1
            )
            # and epoch_ix > 0
        ):
            # print(f"Running validation evaluation at epoch {epoch_ix}")
            
            # Run evaluation on single validation batch
            agent.train(False)  # Set to evaluation mode
            
            with torch.no_grad():
                # Get one validation batch
                val_batch = next(iter(val_loader))
                _, eval_metrics = run_eval(agent=agent, vis=False, batch=val_batch, history_bid=-1)
            
            # print(f"Validation completed. Metrics: {eval_metrics}")
            
            # Log validation metrics
            if cfg.use_wandb:
                # Log regular metrics
                wandb.log(
                    {"val/" + k: v for k, v in eval_metrics.items() if not k.endswith('image')},
                    step=global_step,
                )
                
                # Log validation metadata
                wandb.log({
                    "val/epoch": epoch_ix
                }, step=global_step)
                
                # Log rendered images from validation batch (if any)
                for k, v in eval_metrics.items():
                    if k.endswith('image') and v is not None:
                        # Convert normalized image (0-1) to uint8 (0-255) for wandb
                        if v.dtype == np.float32 and v.max() <= 1.0:
                            v_uint8 = (v * 255).astype(np.uint8)
                        else:
                            v_uint8 = v.astype(np.uint8)
                        
                        wandb.log(
                            {f"val/{k}": wandb.Image(v_uint8)},
                            step=global_step,
                        )
            
            agent.train(True)  # Set back to training mode

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
        # agent.save_snapshot(None)
        # print(f"save ckpt at {save_path}")


if __name__ == "__main__":
    main()
