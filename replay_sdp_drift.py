"""Offline replay for diagnosing SDP inference drift.

Workflow:
  1. Run the live TAMP pipeline once with `EQUIBOT_DUMP_INPUT=/tmp/sdp_snapshot.pkl`
     set; the first `gen_objcentric_traj` call writes the snapshot.
  2. Run this script:
        python replay_sdp_drift.py --snapshot /tmp/sdp_snapshot.pkl --n 5

It rebuilds the wrapper from the checkpoint recorded in the snapshot, then runs
inference N times under several cumulative determinism settings and prints a
table of max-diffs (rotation vs translation broken out for SE(3) keys). The
first row whose rotation diff collapses identifies the residual non-determinism
source after the user's existing `torch.manual_seed(seed)` reset in
`pred_unimanual_traj`.
"""

import argparse
import os
import pickle

import numpy as np
import torch


def load_snapshot(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    print(f"[snapshot] keys={list(payload.keys())}")
    for k, v in payload["agent_obs"].items():
        shape = tuple(v.shape) if torch.is_tensor(v) else getattr(v, "shape", None)
        print(f"[snapshot]   agent_obs[{k!r}] shape={shape}")
    print(f"[snapshot] skill_name={payload['skill_name']!r} task_name={payload['task_name']!r} seed={payload['seed']}")
    print(f"[snapshot] ckpt_path={payload['ckpt_path']!r}")
    return payload


def build_wrapper(payload):
    from equibot.policies.aloha_wrapper import pddl_wrapper
    return pddl_wrapper(
        payload["dataset_path"],
        ckpt_path=payload["ckpt_path"],
        exe_mode="inference",
    )


def _walk_pools(encoder):
    """Yield every pool module on the encoder that exposes `random_start`."""
    if not hasattr(encoder, "down_blocks"):
        return
    for block in encoder.down_blocks:
        pool = block["pool"] if isinstance(block, dict) else getattr(block, "pool", None)
        if pool is not None and hasattr(pool, "random_start"):
            yield pool


def set_encoder_deterministic(wrapper, deterministic):
    enc = wrapper.agent.actor.nets["obj_encoder"]
    enc.deterministic = deterministic
    flipped = []
    for pool in _walk_pools(enc):
        pool.random_start = (not deterministic)
        flipped.append(type(pool).__name__)
    print(f"[knob] encoder.deterministic={deterministic}  pools_flipped={flipped}")


def install_probes(wrapper, recorder):
    """Monkey-patch the policy + encoder forward to record intermediate signals.

    recorder is a list; each inference call appends a dict with:
        obs_vec_sum, noisy_sample (cpu), fps0_indices (cpu)
    """
    actor = wrapper.agent.actor
    encoder = actor.nets["obj_encoder"]

    # Probe FPS-selected indices on the first down-block pool.
    first_pool = None
    for p in _walk_pools(encoder):
        first_pool = p
        break

    if first_pool is not None and not getattr(first_pool, "_probed", False):
        orig_pool_forward = first_pool.forward

        def patched_pool_forward(*args, **kwargs):
            out = orig_pool_forward(*args, **kwargs)
            # Try to grab the FPS-selected node indices from the output. The
            # exact return signature varies between pool types; record whatever
            # tensor is shortest (most likely the index list).
            try:
                idx_like = None
                if isinstance(out, tuple):
                    # heuristic: the indices tensor is 1-D and has the smallest
                    # numel among the returned tensors.
                    candidates = [x for x in out if torch.is_tensor(x) and x.dim() == 1]
                    if candidates:
                        idx_like = min(candidates, key=lambda x: x.numel())
                if idx_like is not None and recorder:
                    recorder[-1]["fps0_indices"] = idx_like.detach().cpu().clone()
            except Exception:
                pass
            return out

        first_pool.forward = patched_pool_forward
        first_pool._probed = True

    # Probe pred_unimanual_traj to capture obs_vec and noisy_sample.
    if not getattr(actor, "_probed", False):
        orig_proc_pc = actor.proc_pc
        def patched_proc_pc(*args, **kwargs):
            obs_vec, center, scale = orig_proc_pc(*args, **kwargs)
            if recorder:
                recorder[-1]["obs_vec_sum"] = float(obs_vec.detach().sum().item())
                recorder[-1]["center"] = center.detach().cpu().clone()
            return obs_vec, center, scale
        actor.proc_pc = patched_proc_pc

        # Hook noise_scheduler.set_timesteps as a per-call phase marker.
        orig_set_timesteps = actor.noise_scheduler.set_timesteps
        def patched_set_timesteps(n, *a, **kw):
            out = orig_set_timesteps(n, *a, **kw)
            if recorder:
                recorder[-1]["call_phase"] = "after_set_timesteps"
            return out
        actor.noise_scheduler.set_timesteps = patched_set_timesteps

        actor._probed = True


def run_n_calls(wrapper, payload, n, np_seed_each_call=False):
    from equibot.policies.utils.misc import to_tensor
    outs = []
    records = []
    for _ in range(n):
        records.append({})  # one slot per call; probes write into records[-1]
        if np_seed_each_call and payload["seed"] is not None:
            np.random.seed(payload["seed"])
        # agent_obs was pickled as CPU tensors; gen_objcentric_traj re-moves to device.
        action_w = wrapper.gen_objcentric_traj(
            payload["obs_key"],
            to_tensor(payload["agent_obs"]),
            skill_name=payload["skill_name"],
            task_name=payload["task_name"],
            seed=payload["seed"],
        )
        outs.append(action_w)
    return outs, records


def report_diffs(outs, label, records=None):
    keys = list(outs[0].keys())
    print(f"\n===== {label} =====")
    for k in keys:
        arr0 = np.asarray(outs[0][k])
        if arr0.ndim >= 2 and arr0.shape[-2:] == (4, 4):
            rot0 = arr0[..., :3, :3]
            tr0 = arr0[..., :3, 3]
            rot_diffs = [np.max(np.abs(np.asarray(o[k])[..., :3, :3] - rot0)) for o in outs[1:]]
            tr_diffs = [np.max(np.abs(np.asarray(o[k])[..., :3, 3] - tr0)) for o in outs[1:]]
            print(f"  {k:24s}  rot_max={max(rot_diffs):.3e}  trans_max={max(tr_diffs):.3e}")
        else:
            diffs = [np.max(np.abs(np.asarray(o[k]) - arr0)) for o in outs[1:]]
            print(f"  {k:24s}  max_diff={max(diffs):.3e}")

    if records:
        obs_vec_sums = [r.get("obs_vec_sum") for r in records if "obs_vec_sum" in r]
        if obs_vec_sums:
            spread = max(obs_vec_sums) - min(obs_vec_sums)
            print(f"  [probe] obs_vec.sum spread across {len(obs_vec_sums)} calls: {spread:.3e}")
        fps0 = [r.get("fps0_indices") for r in records if "fps0_indices" in r]
        if len(fps0) >= 2:
            same = all(torch.equal(fps0[0], x) for x in fps0[1:])
            print(f"  [probe] FPS first-block indices identical across calls: {same}")


def apply_knobs(wrapper, knobs):
    """Apply determinism knobs in-place. knobs is a set of strings."""
    if "torch_global" in knobs:
        seed = 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if "cudnn_det" in knobs:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if "use_det_algos" in knobs:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    if "encoder_det" in knobs:
        set_encoder_deterministic(wrapper, True)
    else:
        set_encoder_deterministic(wrapper, False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, help="Path to pickled gen_objcentric_traj input.")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--ckpt-override", default=None, help="Override ckpt_path stored in snapshot.")
    p.add_argument("--dataset-override", default=None, help="Override dataset_path stored in snapshot.")
    args = p.parse_args()

    payload = load_snapshot(args.snapshot)
    if args.ckpt_override:
        payload["ckpt_path"] = args.ckpt_override
    if args.dataset_override:
        payload["dataset_path"] = args.dataset_override

    wrapper = build_wrapper(payload)
    recorder = []
    install_probes(wrapper, recorder)

    # Cumulative knob plan: each row adds one knob to the previous set.
    plan = [
        ("baseline", set()),
        ("+torch_global", {"torch_global"}),
        ("+cudnn_det", {"torch_global", "cudnn_det"}),
        ("+use_det_algos", {"torch_global", "cudnn_det", "use_det_algos"}),
        ("+encoder_det", {"torch_global", "cudnn_det", "use_det_algos", "encoder_det"}),
        ("+np_seed_each_call", {"torch_global", "cudnn_det", "use_det_algos", "encoder_det"}),
    ]

    for label, knobs in plan:
        apply_knobs(wrapper, knobs)
        recorder.clear()
        outs, records = run_n_calls(
            wrapper, payload, args.n,
            np_seed_each_call=(label == "+np_seed_each_call"),
        )
        report_diffs(outs, label, records=records)

    print("\nDone. The first label whose rot_max ≤ 1e-5 identifies the residual non-determinism source.")


if __name__ == "__main__":
    main()
