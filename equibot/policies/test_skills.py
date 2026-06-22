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




from equibot.policies.utils.misc import  EQUIBOT_PATH, get_agent, get_dataset, vis_metric_imgs



def run_eval(
    agent,
    vis=False,
    log_dir=None,
    use_wandb=False,
    batch = None,
    history_bid = -1,
):

    # ## input obs from dataset
    # agent_obs = get_obs(batch)

    # predict actions
    st = time.time()
    unnormed_history, metrics = agent.eval_with_rotation(batch, history_bid)
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


@hydra.main(config_path="configs", config_name="dmg_threading_per_skill")
def main(cfg):
    cfg.mode = "eval"
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
    # cfg.data.dataset.path=os.path.join(EQUIBOT_PATH, 'data/mj_peg_hole/')
    from equibot.policies.utils.misc import collate_fn
    eval_dataset = get_dataset(cfg, "test")
    num_workers = cfg.data.dataset.num_workers
    test_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=32,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    data_iter = iter(test_loader)
    fist_batch = next(data_iter)

    agent = get_agent(cfg.agent.agent_name)(cfg)
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
            # history_bid = cfg.eval.history_bid,
        )

        print("evaluation result: ", eval_metrics)




if __name__ == "__main__":
    main()
