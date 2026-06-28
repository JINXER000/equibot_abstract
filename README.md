# Per-Skill SIM(3)-Equivariant Diffusion Policy (Threading & Assembly)

A SIM(3)-equivariant diffusion policy that predicts manipulation skills **per skill**
(one conditional model shared across skills, conditioned on a skill/task language
embedding) from point-cloud observations. This release focuses on two bimanual
[DexMimicGen](https://dexmimicgen.github.io/) tasks:

| Config | Task | Skill |
| --- | --- | --- |
| `dmg_threading_per_skill` | `dmg_threading` | per-skill grasp |
| `dmg_assembly_per_skill`  | `dmg_assembly`  | per-skill grasp |

Both configs use the `per_skill` agent (`EquiSkillAgent`) with a frozen VDGCNN
encoder and an SO(3)-equivariant conditional U-Net denoiser. The bimanual
keypose path (`BiopSkillPolicy` / `per_skill_biop_jpose`) is retained for
extension but is not wired to either default config.

## Installation

Tested on Ubuntu 22.04, CUDA 11.8, RTX 4090. From the repository root:

```bash
conda create -n equiv_primitive python=3.9 -y
conda activate equiv_primitive
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
# pytorch3d (used by the VDGCNN encoder and pose estimation) — install from source:
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -e .
```

`pip install -e .` pulls the remaining dependencies (including `transformers`
and `zarr`). `pretrained/chairs.pt` (referenced by the configs as the VDGCNN
backbone `preload_path`) ships with the repository.

Evaluation additionally requires [`robosuite`](https://github.com/ARISE-Initiative/robosuite)
and the DexMimicGen environment/dataset, which are external to this repository.

## Data layout

Each task reads HDF5 demonstrations from `data.dataset.path` and caches BERT
skill/task embeddings under `data.dataset.embedding_cache_dir` (default
`./data/bert`). Point to your generated demonstrations via the `data.dataset.path`
override shown below.

## Training

Training is driven by Hydra; pass one of the two configs by name.

```bash
# threading, per-skill grasp
python -m equiv_primitive.policies.train_skills --config-name dmg_threading_per_skill \
    prefix=dmg_threading \
    data.dataset.path=/path/to/data/dmg_threading/

# assembly, per-skill grasp
python -m equiv_primitive.policies.train_skills --config-name dmg_assembly_per_skill \
    prefix=dmg_assembly \
    data.dataset.path=/path/to/data/dmg_assembly/
```

Set the wandb entity/project at the bottom of
[`equiv_primitive/policies/configs/base.yaml`](equiv_primitive/policies/configs/base.yaml), or pass
`use_wandb=false` to disable logging.

## Evaluation

`eval_dmg_perskill.py` measures per-skill grasp error against object-pose-derived
ground truth by spawning the threading env in RoboSuite. It requires the external
robosuite/DexMimicGen setup.

```bash
python -m equiv_primitive.policies.eval_dmg_perskill \
    --ckpt_glob "logs/train/dmg_threading/*.pth" \
    --K 30 --n_ref 5 --seed 0
```

## Design notes — metric-aligned auxiliary loss

The configs expose a metric-aligned auxiliary loss on top of the standard ε-MSE
diffusion objective (`lambda_pos`, `lambda_rot`, `rot_loss_type`, `snr_gamma`).
It is complementary, not a replacement:

- **ε-MSE is the actual diffusion objective.** The reverse process integrates the
  predicted noise at every timestep, so the rotation channels must be supervised
  at all noise levels for sampling to work.
- **The geodesic/chordal aux loss is deliberately low-`t`.** It sharpens the
  reconstructed *clean* rotation (where final precision is decided) and is
  down-weighted at high noise by min-SNR weighting.
- **rot6d → matrix is many-to-one**, so the geodesic loss alone under-determines
  the 6D vector; only the ε-MSE constrains the full orthonormal structure the
  forward process assumes.

Set `lambda_pos=0` / `lambda_rot=0` to recover the plain ε-MSE training.

## Acknowledgements

This codebase builds on **EquiBot** (Yang\*, Cao\*, Deng, Antonova, Song, Bohg;
[paper](https://arxiv.org/abs/2407.01479), [project](https://equi-bot.github.io)).

## License

MIT License (see [LICENSE](LICENSE)).
