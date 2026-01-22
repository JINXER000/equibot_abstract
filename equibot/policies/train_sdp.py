"""
Training script for Spherical Diffusion Policy (SDP).

Usage:
    python train_sdp.py --config-name sdp_per_skill \
        prefix=sdp_experiment \
        data.dataset.path=/path/to/data \
        use_wandb=true

Reference: paper_ref/sdp/example_paper.tex
"""

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

root_path = next(('/'+ os.path.join(*os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:i+1]) + os.sep
                    for i in range(len(os.path.dirname(os.path.abspath(__file__)).split(os.sep))) 
                    if os.path.dirname(os.path.abspath(__file__)).split(os.sep)[i] == 'equibot_abstract'), None)
import sys
sys.path.append(root_path) if root_path not in sys.path else None

from equibot.policies.utils.misc import EQUIBOT_PATH, get_agent, get_dataset

try:
    from .test_sdp import run_eval
except ImportError:
    from test_sdp import run_eval


@hydra.main(config_path=os.path.join(EQUIBOT_PATH, "equibot/policies/configs"), config_name="sdp_per_skill")
def main(cfg):
    """
    Main training loop for SDP.
    
    Training objective (from paper):
    L = ||epsilon_theta(S_t, A_t^0 + epsilon, k) - epsilon||^2
    
    Where:
    - S_t: State (point cloud encoded by EquiformerV2)
    - A_t^0: Clean action trajectory
    - epsilon: Gaussian noise
    - k: Denoising step
    """
    assert cfg.mode == "train"
    np.random.seed(cfg.seed)
    
    # Initialize parameters
    batch_size = cfg.training.batch_size
    
    # Load the full dataset (reuses per_skill_dataset.py format)
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
    
    # Import collate function
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
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    cfg.data.dataset.num_training_steps = (
        cfg.training.num_epochs * len(train_dataset) // batch_size
    )
    
    # Initialize SDP agent
    agent = get_agent(cfg.agent.agent_name)(cfg)
    
    # Load checkpoint if specified
    if cfg.training.ckpt is not None:
        agent.load_snapshot(cfg.training.ckpt)
        start_epoch_ix = int(cfg.training.ckpt.split("/")[-1].split(".")[0][4:])
    else:
        start_epoch_ix = 0
    
    # Copy normalizer from dataset
    agent.set_normalizer_and_statistics(full_dataset)
    
    # WandB logging
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
            tags=["train", "sdp"],
            name=cfg.prefix,
            settings=wandb.Settings(code_dir="."),
            config=wandb_config,
        )
    else:
        log_dir = None
    
    # Training loop
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
        
        # Run validation evaluation
        if (
            epoch_ix % cfg.training.eval_interval == 0
            or epoch_ix == cfg.training.num_epochs - 1
        ):
            agent.train(False)
            
            with torch.no_grad():
                val_batch = next(iter(val_loader))
                _, eval_metrics = run_eval(agent=agent, vis=False, batch=val_batch, history_bid=-1)
            
            if cfg.use_wandb:
                wandb.log(
                    {"val/" + k: v for k, v in eval_metrics.items() if not k.endswith('image')},
                    step=global_step,
                )
                wandb.log({"val/epoch": epoch_ix}, step=global_step)
                
                # Log rendered images
                for k, v in eval_metrics.items():
                    if k.endswith('image') and v is not None:
                        if v.dtype == np.float32 and v.max() <= 1.0:
                            v_uint8 = (v * 255).astype(np.uint8)
                        else:
                            v_uint8 = v.astype(np.uint8)
                        wandb.log(
                            {f"val/{k}": wandb.Image(v_uint8)},
                            step=global_step,
                        )
            
            agent.train(True)
        
        # Save checkpoint
        if log_dir is not None and (
            epoch_ix % cfg.training.save_interval == 0
            or epoch_ix == cfg.training.num_epochs - 1
        ):
            save_path = os.path.join(log_dir, f"ckpt{epoch_ix:05d}.pth")
            num_ckpt_to_keep = 1
            if len(list(glob(os.path.join(log_dir, "ckpt*.pth")))) > num_ckpt_to_keep:
                for fn in list(sorted(glob(os.path.join(log_dir, "ckpt*.pth"))))[:-num_ckpt_to_keep]:
                    os.remove(fn)
            agent.save_snapshot(save_path)


if __name__ == "__main__":
    main()

