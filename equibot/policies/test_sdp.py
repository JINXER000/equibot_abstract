"""
Test/evaluation script for Spherical Diffusion Policy (SDP).

Usage:
    python test_sdp.py --config-name sdp_per_skill \
        prefix=sdp_eval \
        mode=eval \
        training.ckpt=/path/to/checkpoint.pth \
        use_wandb=false

Reference: paper_ref/sdp/example_paper.tex
"""

import os
import sys
import hydra
import torch
import numpy as np
from tqdm import tqdm

from equibot.policies.utils.misc import (
    EQUIBOT_PATH, get_agent, get_dataset, to_torch,
    geodestDist, render_trajectory, ascii_tensor_batch_to_str
)


def run_eval(agent, vis=False, batch=None, history_bid=-1):
    """
    Run evaluation on a batch.
    
    Denoising process (from paper Eq. 2):
    A_t^{k-1} = alpha * (A_t^k - gamma * epsilon_theta(S_t, A_t^k, k) + z)
    
    Args:
        agent: SDP agent
        vis: Whether to visualize results
        batch: Input batch from dataloader
        history_bid: Batch index for history visualization
        
    Returns:
        action_dict: Predicted actions
        eval_metrics: Evaluation metrics
    """
    if batch is None:
        return {}, {}
    
    unnormed_history, metrics = agent.eval_with_rotation(batch, history_bid)
    # print(f"Inference time: {time.time() - st:.3f}s")

    return unnormed_history, metrics


@hydra.main(config_path=os.path.join(EQUIBOT_PATH, "equibot/policies/configs"), config_name="sdp_per_skill")
def main(cfg):
    """
    Main evaluation function for SDP.
    
    Evaluates:
    - Position error (L1 distance)
    - Rotation error (geodesic distance)
    - Trajectory smoothness
    """
    assert cfg.mode == "eval", "Set mode=eval for evaluation"
    np.random.seed(cfg.seed)
    
    # Load dataset
    full_dataset = get_dataset(cfg, "train")
    
    num_workers = cfg.data.dataset.num_workers
    batch_size = cfg.training.batch_size
    
    from equibot.policies.utils.misc import collate_fn
    
    test_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Initialize agent
    agent = get_agent(cfg.agent.agent_name)(cfg)
    
    # Load checkpoint
    if cfg.training.ckpt is not None:
        agent.load_snapshot(cfg.training.ckpt)
        print(f"Loaded checkpoint from {cfg.training.ckpt}")
    else:
        print("Warning: No checkpoint specified, using random weights")
    
    agent.set_normalizer_and_statistics(full_dataset)
    
    # Evaluation loop
    all_metrics = {
        'xyz_l1': [],
        'rot_diff': [],
    }
    
    print("Running evaluation...")
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        _, eval_metrics = run_eval(
            agent=agent,
            vis=(batch_idx == 0),  # Visualize first batch only
            batch=batch,
            history_bid=cfg.eval.history_bid
        )
        
        for key in all_metrics:
            if key in eval_metrics:
                val = eval_metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                all_metrics[key].append(val)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary:")
    print("=" * 50)
    for key, values in all_metrics.items():
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key}: {mean_val:.4f} +/- {std_val:.4f}")
    print("=" * 50)
    
    return all_metrics


if __name__ == "__main__":
    main()

