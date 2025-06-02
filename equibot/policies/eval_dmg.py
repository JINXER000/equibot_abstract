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
from equibot.policies.utils.misc import  EQUIBOT_PATH, get_agent, get_dataset





@hydra.main(config_path="configs", config_name="mj_peg_hole")
def main(cfg):
    cfg.mode == "eval"
    device = torch.device(cfg.device)

    agent = get_agent(cfg.agent.agent_name)(cfg)
    agent.train(False)
    ckpt_full_path = os.path.join(EQUIBOT_PATH, cfg.training.ckpt)
    agent.load_snapshot(ckpt_full_path)

    dmg_dir = cfg.eval.dmg_dir
    sys.path.append(dmg_dir)
    from scripts.eval_dmg_wrapper import DMG_env_runner

    task_name = 'two_arm_three_piece_assembly'
    env_runner = DMG_env_runner(task_name)
    dataset_path = cfg.data.dataset.path
    obs = env_runner.rollout_from_biop(dataset_path)

    env_runner.get_equipolicy_agent(agent, equi_cfg=cfg)




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

    metrics = env_runner.evaluate_fn(obs)
    # obs_history = [obs]
    # done = False
    # while not done:
    #     new_obs, done, metrics = env_runner.evaluate_fn(obs)
    #     obs_history.append(new_obs)
    #     obs = new_obs

if __name__ == "__main__":
    main()