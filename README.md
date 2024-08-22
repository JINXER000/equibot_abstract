# CYZ dev


## TODO
### problems and solution
[x] use grasp pose relative to obj. before normalizing pc, subtract the pc with pc center. --> NO need, as when producing pc_feat, it already subtracted the centroid.
[x] Use grasp data for riemann to train. --> still rot incorrect
[x] Visualize the grasp diffusion history. 
[x] feat_dict["so3"] and feat_dict["inv"] seems to be so small. Is it because of nomalizer?  --> It becomes better with more data. 
[x] Make grasp_xyz normalizer different --> still incorrect
[x] perform orthogonization on rotation mat
[] find out why joint error is big in evaluation. Should we use 6 dof only? Still no use. Should we use txt joint data?
[] Try if training joints and set grasp as rand is OK?


### representative runs
- when using only one grasp and point cloud, the rotation error is large:
https://wandb.ai/neuralogic/equibot/runs/bdwdh2iv?nw=nwuserjosephchen

- when using 50 demos, we observe that at 50k step,  the encoder output goes up, and rotation error drops as well!
https://wandb.ai/neuralogic/equibot/runs/33c0qst9?nw=nwuserjosephchen

- however, when training grasp + jpose, jpose is not accurate...
https://wandb.ai/neuralogic/equibot/runs/c9us76fl?nw=nwuserjosephchen

- Using 2 layer encoder, output 1 grasp + 6 joints, worse..
https://wandb.ai/neuralogic/equibot/runs/1evwj954?nw=nwuserjosephchen

#### questions about convergence
Is it because of the weight initialization of the encoder? we need more point clouds?
I found that vec_layer use kaiming initialization. 
I think in aloha case, we should use 2-layer encoder instead of 4-layer. 
### compose 2 grasps
#### in training, 

- use 2 unet to sum the loss.  Note that each grasp should subtract the corresponding point cloud. As a result, the right grasp should be normal distribution around the right point cloud.
- in CCSP, it average outputs from different constraints(mlps). I think it is also doable, as the right grasp pred output is not taken into account in the left-unet. 
- specifically, we have vec_lgrasp_pred and vec_rgrasp_pred, then we concatnate them to compute the loss. For joint, we calculate the average output from 2 unets. 
- I think it can jointly model the moultimodality of 2 grasps. For example, left grasp on top + right grasp on bottom, it will be in the same pred_vec.
#### in testing, 
refer to ECCV 2022 to sample the joint poses. For grasps, sample them individually. 

### use MLP instead of Unet as denoising network

## Adapt to grasp+joint
### eef data structure
[x] figure out data structure of orientation. 
The rotation is the 1st and 3rd row of rot mat. See trans2vec() for more detail.
### Procedure to revise the state space
[x] train the grasp pose and joint vals
- in aloha_policy.py, remember to revise self.eef_dim. 
- revise _init_normalizers() if the input is mixed with scalars and vectors. 

# EquiBot: SIM(3)-Equivariant Diffusion Policy for Generalizable and Data Efficient Learning

Jingyun Yang*, Zi-ang Cao*, Congyue Deng, Rika Antonova, Shuran Song, Jeannette Bohg

<a href='https://equi-bot.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2407.01479'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/FFrl_TEXrUw)

![Overview figure](https://equi-bot.github.io/images/teaser.jpg)

This repository includes:

* Implementation of the EquiBot method and a Diffusion Policy baseline that takes point clouds as input.
* A set of three simulated mobile manipulation environments: Cloth Folding, Object Covering, and Box Closing.
* Data generation, training, and evaluation scripts that accompany the above algorithms and environments.

## Getting Started

### Installation

This codebase is tested with the following setup: Ubuntu 20.04, an RTX 4090 GPU, CUDA 11.8. In the root directory of the repository, run the following commands:

```
conda create -n lfd python=3.10 -y
conda activate lfd

conda install -y fvcore iopath ffmpeg -c iopath -c fvcore
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install -e .
```

Then, in the last two lines of [this config file](equibot/policies/configs/base.yaml), enter the wandb entity and project names for logging purposes. If you do not have a wandb account yet, you can register [here](https://wandb.ai).

### Demonstration Generation

The following code generates demonstrations for simulated mobile environments. To change the number of generated demos, change `--num_demos 50` to a different number.

```
python -m equibot.envs.sim_mobile.generate_demos --data_out_dir ../data/fold \
    --num_demos 50 --cam_dist 2 --cam_pitches -75 --task_name fold

python -m equibot.envs.sim_mobile.generate_demos --data_out_dir ../data/cover \
    --num_demos 50 --cam_dist 2 --cam_pitches -75 --task_name cover

python -m equibot.envs.sim_mobile.generate_demos --data_out_dir ../data/close \
    --num_demos 50 --cam_dist 1.5 --cam_pitches -45 --task_name close
```

### Training

The following code runs training for our method and the Diffusion Policy baseline. Fill the dataset path with the `data_out_dir` argument in the previous section. Make sure the dataset path ends with `pcs`. To run this code for the `cover` and `close` environments, substitute occurrences of `fold` with `cover` or `close`.

```
# diffusion policy baseline (takes point clouds as input)
python -m equibot.policies.train --config-name fold_mobile_dp \
    prefix=sim_mobile_fold_7dof_dp \
    data.dataset.path=[data out dir in the last section]/pcs

# our method (equibot)
python -m equibot.policies.train --config-name fold_mobile_equibot \
    prefix=sim_mobile_fold_7dof_equibot \
    data.dataset.path=[data out dir in the last section]/pcs
```
in my case, I can use the command below for training:



```
python -m equibot.policies.train --config-name fold_mobile_equibot \
    prefix=sim_mobile_fold_7dof_equibot \
    data.dataset.path=/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/data/fold/pcs/
```

```
cd equibot/policies/
python train_abstract.py --config-name transfer_tape \
    prefix=aloha_transfer_tape \
    data.dataset.path=/home/user/yzchen_ws/docker_share_folder/difussion/equibot_abstract/data/transfer_tape/
```


### Evaluation

The commands below evaluate the trained EquiBot policy on the four different setups mentioned in the paper: `Original`, `R+Su`, `R+Sn`, and `R+Sn+P`. To run these evaluations for the DP baseline, replace all occurrences of `equibot` to`dp`. For the log directory, fill `[log_dir]` with the absolute path to the log directory. By default, this directory is `./log`.

```
# Original setup
python -m equibot.policies.eval --config-name fold_mobile_equibot \
    prefix="eval_original_sim_mobile_fold_equibot_s1" mode=eval \
    training.ckpt="[log_dir]/train/sim_mobile_fold_7dof_equibot_s1/ckpt01999.pth" \
    env.args.max_episode_length=50 env.vectorize=true

# R+Su setup
python -m equibot.policies.eval --config-name fold_mobile_equibot \
    prefix="eval_rsu_sim_mobile_fold_7dof_equibot_s1" mode=eval \
    training.ckpt="[log_dir]/train/sim_mobile_fold_7dof_equibot_s1/ckpt01999.pth" \
    env.args.scale_high=2 env.args.uniform_scaling=true \
    env.args.randomize_rotation=true env.args.randomize_scale=true env.vectorize=true

# R+Sn setup
python -m equibot.policies.eval --config-name fold_mobile_equibot \
    prefix="eval_rsn_sim_mobile_fold_7dof_equibot_s1" mode=eval \
    training.ckpt="[log_dir]/train/sim_mobile_fold_7dof_equibot_s1/ckpt01999.pth" \
    env.args.scale_high=2 env.args.scale_aspect_limit=1.33 \
    env.args.randomize_rotation=true env.args.randomize_scale=true env.vectorize=true

# R+Sn+P setup
python -m equibot.policies.eval --config-name fold_mobile_equibot \
    prefix="eval_rsnp_sim_mobile_fold_7dof_equibot_s1" mode=eval \
    training.ckpt="[log_dir]/train/sim_mobile_fold_7dof_equibot_s1/ckpt01999.pth" \
    env.args.scale_high=2 env.args.scale_aspect_limit=1.33 \
    env.args.randomize_rotation=true env.args.randomize_scale=true \
    +env.args.randomize_position=true +env.args.rand_pos_scale=0.5 env.vectorize=true
```

## License

This codebase is licensed under the terms of the MIT License.
