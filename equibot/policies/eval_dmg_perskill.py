"""Offline per-skill grasp-prediction error eval on the DexMimicGen threading sim.

The ``per_skill`` checkpoints under ``logs/train/dmg_threading/200demo*.pth`` only
learned the two grasp skills (``robot0_grasp_tripod_obj`` / ``robot1_grasp_needle_obj``);
they have no bimanual skill, so full threading success is not measurable. This script
therefore reports *grasp-prediction error* against an object-pose-derived ground truth:

    1. Build a reference grasp trajectory in each object's body frame, once, from a few
       demos:  T_ref_in_obj = inv(T_obj_demo) @ T_grasp_demo_world.
    2. For K fresh env spawns (randomized object placement), render the object's point
       cloud from the live sim, run per-skill inference, and compare the predicted
       world-frame keyposes to  T_gt_world = T_obj_cur @ T_ref_in_obj.
    3. Report mean position error (m) and rotation error (deg) per checkpoint / skill.

Run inside the ``sdp_dmg`` conda env (robosuite 1.5.1).  Example::

    MUJOCO_GL=egl python -m equibot.policies.eval_dmg_perskill --K 30
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import glob
import argparse
from collections import defaultdict

import numpy as np

DEXMIMICGEN_PATH = "/home/user/yzchen_ws/imitation_learning/dexmimicgen"
if DEXMIMICGEN_PATH not in sys.path:
    sys.path.insert(0, DEXMIMICGEN_PATH)

import h5py
import mujoco
import robosuite

from scripts.playback_depth import (
    D_CAM,
    get_env_metadata_from_dataset,
    get_pcd_dict_fn,
    reset_to,
)

from equibot.policies.aloha_wrapper import pddl_wrapper
from equibot.policies.utils.misc import choose_ids, compose_transformation

TASK_NAME = "two_arm_threading"
TRAJ_LEN = 8
PC_HW = 128
INTERESTED_OBJS = ["needle_obj", "tripod_obj"]
# skill key -> (conditioning object, demonstrating robot)
SKILL_SPECS = {
    "robot0_grasp_tripod_obj": ("tripod_obj", "robot0"),
    "robot1_grasp_needle_obj": ("needle_obj", "robot1"),
}


# --------------------------------------------------------------------------- env

def build_env(hdf5_path):
    """Instantiate the threading env with depth + instance-segmentation cameras."""
    meta = get_env_metadata_from_dataset(dataset_path=hdf5_path)
    env_kwargs = dict(meta["env_kwargs"])
    env_kwargs["env_name"] = meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_depths"] = True
    env_kwargs["camera_segmentations"] = "instance"
    env_kwargs["camera_names"] = D_CAM
    env_kwargs["camera_heights"] = PC_HW
    env_kwargs["camera_widths"] = PC_HW
    env_kwargs.pop("env_lang", None)
    return robosuite.make(**env_kwargs)


def read_object_pose(env, obj_name):
    """World-frame SE(3) pose (4, 4) of an object, read from its mujoco root body."""
    model = env.sim.model._model
    data = env.sim.data._data
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{obj_name}_root")
    if body_id < 0:
        raise KeyError(f"mujoco body not found: {obj_name}_root")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = data.xpos[body_id]
    pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
    return pose


# ------------------------------------------------------------------- references

def demo_grasp_world(demo, skill_name, rbt_name, ref_seed):
    """Ground-truth grasp keyposes (T, 4, 4) of one demo, world frame.

    Mirrors ``PerSkillDataset.get_dataslice_unimanual`` keypose selection. The
    ``choose_ids`` sampler is stochastic, so the seed is fixed for reproducibility.
    """
    skill_grp = demo[f"sg_info/{skill_name}"]
    extended_ids = skill_grp["extended_ids"][()]
    essential_ids = skill_grp["essential_ids"][()]

    np.random.seed(ref_seed)
    ids = choose_ids(TRAJ_LEN, extended_ids, essential_ids, "grasp")

    eef_pos = demo[f"obs/{rbt_name}_eef_pos"][()][ids]
    eef_quat = demo[f"obs/{rbt_name}_eef_quat"][()][ids]
    return np.stack(list(map(compose_transformation, eef_pos, eef_quat)), axis=0)


def average_se3(transforms):
    """Mean of a stack (M, T, 4, 4) of SE(3) trajectories -> (T, 4, 4).

    Translations are averaged directly; rotations are averaged then re-projected
    onto SO(3) via SVD (valid because per-skill grasps are near-identical).
    """
    transforms = np.asarray(transforms, dtype=np.float64)
    out = np.tile(np.eye(4), (transforms.shape[1], 1, 1))
    out[:, :3, 3] = transforms[:, :, :3, 3].mean(axis=0)
    for t in range(transforms.shape[1]):
        u, _, vt = np.linalg.svd(transforms[:, t, :3, :3].mean(axis=0))
        rot = u @ vt
        if np.linalg.det(rot) < 0:
            u[:, -1] *= -1
            rot = u @ vt
        out[t, :3, :3] = rot
    return out


def build_references(hdf5_path, n_ref):
    """Per-skill reference grasp in object body frame: {skill: (T, 4, 4)}.

    Uses a dedicated env because ``reset_to`` reloads demo-specific scene XML and
    would otherwise contaminate the randomized placement of the eval env.
    """
    env = build_env(hdf5_path)
    refs = {}
    try:
        with h5py.File(hdf5_path, "r") as f:
            demo_names = sorted(
                (k for k in f["data"].keys() if k.startswith("demo_")),
                key=lambda k: int(k.split("_")[-1]),
            )[:n_ref]
            for skill_name, (obj_name, rbt_name) in SKILL_SPECS.items():
                per_demo = []
                for ref_seed, demo_name in enumerate(demo_names):
                    demo = f[f"data/{demo_name}"]
                    reset_to(
                        env,
                        {
                            "states": demo["states"][()][0],
                            "model": demo.attrs["model_file"],
                            "ep_meta": demo.attrs.get("ep_meta", None),
                        },
                    )
                    t_obj = read_object_pose(env, obj_name)
                    t_grasp = demo_grasp_world(demo, skill_name, rbt_name, ref_seed)
                    per_demo.append(np.linalg.inv(t_obj) @ t_grasp)
                refs[skill_name] = average_se3(per_demo)
    finally:
        env.close()
    return refs


# -------------------------------------------------------------------- evaluation

def evaluate_checkpoint(env, pc_fn, refs, ckpt_path, num_spawns, placement_seed, pcd_noise):
    """Mean grasp error per skill for one checkpoint over ``num_spawns`` env resets."""
    wrapper = pddl_wrapper(
        dataset_path="dmg_threading_eval", ckpt_path=ckpt_path, exe_mode="inference"
    )
    errors = defaultdict(lambda: {"pos_err_m": [], "rot_err_deg": []})

    # Seed so every checkpoint sees the identical sequence of object placements.
    np.random.seed(placement_seed)
    for spawn in range(num_spawns):
        obs = env.reset()
        pcd = pc_fn(env, obs)
        for skill_name, (obj_name, _) in SKILL_SPECS.items():
            object_pc = np.asarray(pcd[obj_name].points)
            if object_pc.shape[0] == 0:
                continue
            # Fixed per-spawn diffusion seed -> differences reflect the model, not noise.
            pred = wrapper.predict_skill_keyposes_world(
                object_pc, skill_name, TASK_NAME, seed=spawn, pcd_noise=pcd_noise
            )
            gt = read_object_pose(env, obj_name) @ refs[skill_name]
            err = wrapper.keypose_error(pred["eefpos"], gt)
            errors[skill_name]["pos_err_m"].append(err["pos_err_m"])
            errors[skill_name]["rot_err_deg"].append(err["rot_err_deg"])
    return errors


def print_table(results):
    head = f"{'checkpoint':<28}{'skill':<26}{'N':>4}{'pos_err(m)':>13}{'rot_err(deg)':>14}"
    print("\n" + head)
    print("-" * len(head))
    for ckpt_name, errors in results.items():
        all_pos, all_rot = [], []
        for skill_name in SKILL_SPECS:
            pos = errors[skill_name]["pos_err_m"]
            rot = errors[skill_name]["rot_err_deg"]
            all_pos += pos
            all_rot += rot
            n = len(pos)
            mp = np.mean(pos) if n else float("nan")
            mr = np.mean(rot) if n else float("nan")
            print(f"{ckpt_name:<28}{skill_name:<26}{n:>4}{mp:>13.4f}{mr:>14.3f}")
        n = len(all_pos)
        mp = np.mean(all_pos) if n else float("nan")
        mr = np.mean(all_rot) if n else float("nan")
        print(f"{ckpt_name:<28}{'OVERALL':<26}{n:>4}{mp:>13.4f}{mr:>14.3f}")
        print("-" * len(head))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_glob", default="logs/train/dmg_threading/200demo*.pth")
    parser.add_argument(
        "--hdf5",
        default=os.path.join(
            DEXMIMICGEN_PATH,
            "datasets/generated/two_arm_threading_pc_instance200_sg_200.hdf5",
        ),
    )
    parser.add_argument("--K", type=int, default=30, help="number of env spawns")
    parser.add_argument("--n_ref", type=int, default=5, help="demos for reference grasp")
    parser.add_argument("--seed", type=int, default=0, help="placement seed")
    parser.add_argument(
        "--pcd_noise",
        type=float,
        default=0.0,
        help="std (m) of xyz jitter added to the eval point cloud, matching "
        "training add_pcd_noise (e.g. 0.002); 0 = clean observation",
    )
    args = parser.parse_args()

    ckpt_paths = sorted(glob.glob(args.ckpt_glob))
    if not ckpt_paths:
        raise FileNotFoundError(f"no checkpoints matched: {args.ckpt_glob}")
    print(
        f"Evaluating {len(ckpt_paths)} checkpoint(s) over K={args.K} spawns "
        f"(pcd_noise={args.pcd_noise}):"
    )
    for p in ckpt_paths:
        print(f"  - {p}")

    refs = build_references(args.hdf5, args.n_ref)

    env = build_env(args.hdf5)
    pc_fn = get_pcd_dict_fn(D_CAM, PC_HW, PC_HW, INTERESTED_OBJS)
    results = {}
    try:
        for ckpt_path in ckpt_paths:
            results[os.path.basename(ckpt_path)] = evaluate_checkpoint(
                env, pc_fn, refs, ckpt_path, args.K, args.seed, args.pcd_noise
            )
    finally:
        env.close()

    print_table(results)


if __name__ == "__main__":
    main()
