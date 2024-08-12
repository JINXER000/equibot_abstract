import numpy as np
import torch
from torch import nn

from equibot.policies.utils.norm import Normalizer
from equibot.policies.utils.misc import to_torch
from equibot.policies.utils.diffusion.lr_scheduler import get_scheduler

from equibot.policies.agents.aloha_policy import ALOHAPolicy

class ALOHAAgent(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._init_actor()
        if cfg.mode == "train":
            self.optimizer = torch.optim.AdamW(
                self.actor.nets.parameters(),
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
            )
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_warmup_steps=500,
                num_training_steps=cfg.data.dataset.num_training_steps,
            )
        self.device = cfg.device
        self.num_eef = cfg.env.num_eef
        self.dof = cfg.env.dof
        self.num_points = cfg.data.dataset.num_points
        self.obs_mode = cfg.model.obs_mode
        self.ac_mode = cfg.model.ac_mode
        self.obs_horizon = cfg.model.obs_horizon
        self.pred_horizon = cfg.model.pred_horizon
        self.shuffle_pc = cfg.data.dataset.shuffle_pc

        self.pc_normalizer = None
        self.grasp_xyz_normalizer = None
        self.jpose_normalizer = None

        self.pc_scale = None

        self.symb_mask = cfg.data.dataset.symb_mask


    def _init_actor(self):
        self.actor = ALOHAPolicy(self.cfg, device=self.cfg.device).to(self.cfg.device)
        self.actor.ema.averaged_model.to(self.cfg.device)

    # NOTE: this initialization only on the 1st batch data
    def _init_normalizers(self, batch):
        if self.jpose_normalizer is None: # normalize to [0, max]
            joint_data  = batch["joint_pose"]
            flattend_joint_data = joint_data.view(-1, self.dof)
            indices = [[0, 1, 2, 3, 4, 5]]


            jpose_normalizer = Normalizer(
                flattend_joint_data, symmetric=True, indices=indices
            )
            self.jpose_normalizer = Normalizer(
                {
                    "min": jpose_normalizer.stats["min"],
                    "max": jpose_normalizer.stats["max"],
                }
            )
            print(f"Joint pose normalization stats: {self.jpose_normalizer.stats}")

        if self.grasp_xyz_normalizer is None:
            # grasp_poses = batch['grasp_pose']
            # indices = [[0, 1, 2], [3, 4, 5, 6, 7, 8]] # xyz, rot6d
            # gpose_normalizer = Normalizer(grasp_poses, symmetric=True, indices=indices)
            # self.grasp_xyz_normalizer = Normalizer(
            # {
            #     "min": gpose_normalizer.stats["min"],
            #     "max": gpose_normalizer.stats["max"],
            # }

            # )
            self.grasp_xyz_normalizer = Normalizer(
            {
                "min": jpose_normalizer.stats["min"][:3],
                "max": jpose_normalizer.stats["max"][:3],
            }
            )
        if self.pc_normalizer is None:
            self.pc_normalizer = self.grasp_xyz_normalizer
            self.actor.pc_normalizer = self.pc_normalizer

        # compute action scale relative to point cloud scale
        pc = batch["pc"].reshape(-1, self.num_points, 3)
        centroid = pc.mean(1, keepdim=True)
        centered_pc = pc - centroid
        pc_scale = centered_pc.norm(dim=-1).mean()
        ac_scale = jpose_normalizer.stats["max"].max()
        self.pc_scale = pc_scale / ac_scale
        self.actor.pc_scale = self.pc_scale

    def train(self, training=True):
        self.actor.nets.train(training)

    def update(self, batch, vis=False):
        self.train()

        batch = to_torch(batch, self.device)
        pc = batch["pc"]
        joint_data  = batch["joint_pose"]
        grasp_pose = batch["grasp_pose"]
        pc = pc.repeat(1, self.obs_horizon, 1, 1)
        joint_data = joint_data.repeat(1, self.pred_horizon, 1, 1)
        grasp_pose = grasp_pose.repeat(1, self.pred_horizon, 1, 1)

        if self.pc_scale is None:
            self._init_normalizers(batch)

        # proc pc
        pc = self.pc_normalizer.normalize(pc)
        batch_size = pc.shape[0]
        feat_dict = self.actor.encoder_handle(pc, target_norm=self.pc_scale)
        center = (
            feat_dict["center"].reshape(batch_size, self.obs_horizon, 1, 3)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        )
        scale = feat_dict["scale"].reshape(batch_size, self.obs_horizon, 1, 1)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        
        # proc grasp pose. first split flattened pose to xyz and dir, then normalize xyz. 
        grasp_xyz, grasp_dir1, grasp_dir2 = self.actor._convert_grasp_to_vec(grasp_pose)
        grasp_xyz = self.grasp_xyz_normalizer.normalize(grasp_xyz)
        grasp_xyz = (grasp_xyz - center)/scale
        gt_grasp_z = torch.cat([grasp_xyz, grasp_dir1, grasp_dir2], dim=-2)

        # scalar
        joint_data = self.jpose_normalizer.normalize(joint_data)
        scalar_jpose = self.actor._convert_jpose_to_vec(joint_data)
        # scalar_jpose = joint_data
        
        pc_feat = feat_dict["so3"]  
        obs_cond_vec = pc_feat.reshape(batch_size, -1, 3)

        timesteps = torch.randint(
            0,
            self.actor.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()

        vec_grasp_noise = torch.randn_like(gt_grasp_z, device=self.device)  
        noisy_grasp = self.actor.noise_scheduler.add_noise(
            gt_grasp_z, vec_grasp_noise, timesteps
        )      

        scalar_jpose_noise = torch.randn_like(scalar_jpose, device=self.device)
        noisy_jpose = self.actor.noise_scheduler.add_noise(
            scalar_jpose, scalar_jpose_noise, timesteps
        )

        # only grasp
        if self.symb_mask[0] == 'None' and self.symb_mask[1] == 'None':
            vec_grasp_pred,_  = self.actor.noise_pred_net_handle(
                noisy_grasp,
                timesteps,
                cond=obs_cond_vec,
                scalar_cond=None,
            )
        else:
            # pred joint
            vec_grasp_pred ,  scalar_jpose_pred = self.actor.noise_pred_net_handle(
                noisy_grasp,
                timesteps,
                scalar_sample = noisy_jpose, 
                cond=obs_cond_vec,
                scalar_cond=None,
            )


        # only qpose
        if self.symb_mask[2] == 'None' and self.symb_mask[3] == 'None':
            loss = nn.functional.mse_loss(scalar_jpose_pred, scalar_jpose_noise)
        # only grasp
        elif self.symb_mask[0] == 'None' and self.symb_mask[1] == 'None':
            loss = nn.functional.mse_loss(vec_grasp_pred, vec_grasp_noise)
        else:
            n_vec = np.prod(vec_grasp_pred.shape)
            n_scalar = np.prod(scalar_jpose_pred.shape)
            k = n_vec / (n_vec + n_scalar)
            loss = k * nn.functional.mse_loss(vec_grasp_pred, vec_grasp_noise) \
                + (1 - k) * nn.functional.mse_loss(scalar_jpose_pred, scalar_jpose_noise)
        if torch.isnan(loss):
            print(f"Loss is nan, please investigate.")
            import pdb

            pdb.set_trace()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.actor.step_ema()

        metrics = {"loss": loss,
                   "cond_obsv": np.linalg.norm(
                       obs_cond_vec.detach().cpu().numpy(), axis=1
                   ).mean(),
                   }
        if self.symb_mask[0] != 'None' or self.symb_mask[1] != 'None': # pred joint
            metrics["mean_gt_jnoise_norm"] = np.linalg.norm(
                    scalar_jpose_noise.detach().cpu().numpy(),
                    axis=1,
                ).mean(),
            metrics["mean_pred_jnoise_norm"] = np.linalg.norm(
                    scalar_jpose_pred.detach()
                    .cpu()
                    .numpy(),
                    axis=1,
                ).mean(),
        if self.symb_mask[2] != 'None' or self.symb_mask[3] != 'None': # pred grasp
            metrics["mean_gt_eef_noise_norm"] = np.linalg.norm(
                    vec_grasp_noise.detach().cpu().numpy(), axis=1
                ).mean(),
            metrics["mean_pred_eef_noise_norm"]= np.linalg.norm(
                    vec_grasp_pred.detach().cpu().numpy(), axis=1
                ).mean(),

        return metrics

    def fix_checkpoint_keys(self, state_dict):
        fixed_state_dict = dict()
        for k, v in state_dict.items():
            if "encoder.encoder" in k:
                fixed_k = k.replace("encoder.encoder", "encoder")
            else:
                fixed_k = k
            if "handle" in k:
                continue
            fixed_state_dict[fixed_k] = v
        return fixed_state_dict

    def save_snapshot(self, save_path):
        state_dict = dict(
            actor=self.actor.state_dict(),
            ema_model=self.actor.ema.averaged_model.state_dict(),
            pc_scale=self.pc_scale,
            pc_normalizer=self.pc_normalizer.state_dict(),
            grasp_normalizer=self.grasp_xyz_normalizer.state_dict(),
            ac_normalizer=self.jpose_normalizer.state_dict(),
        )
        torch.save(state_dict, save_path)

    def load_snapshot(self, load_path):
        state_dict = torch.load(load_path)
        self.grasp_xyz_normalizer = Normalizer(state_dict["grasp_normalizer"])
        # self.grasp_xyz_normalizer = Normalizer(state_dict["state_normalizer"])
        self.actor.grasp_xyz_normalizer = self.grasp_xyz_normalizer
        self.jpose_normalizer= Normalizer(state_dict["ac_normalizer"])
        self.actor.jpose_normalizer = self.jpose_normalizer
        if self.obs_mode.startswith("pc"):
            self.pc_normalizer = self.grasp_xyz_normalizer
            self.actor.pc_normalizer = self.pc_normalizer
        del self.actor.encoder_handle
        del self.actor.noise_pred_net_handle
        self.actor.load_state_dict(self.fix_checkpoint_keys(state_dict["actor"]))
        self.actor._init_torch_compile()
        self.actor.ema.averaged_model.load_state_dict(
            self.fix_checkpoint_keys(state_dict["ema_model"])
        )
        self.pc_scale = state_dict["pc_scale"]
        self.actor.pc_scale = self.pc_scale
        


    def act(self, obs, return_history = True):
        self.train(False)
        assert isinstance(obs["pc"][0], np.ndarray)


        # batch_size = len(obs["pc"][0])
        batch_size = 1  # only support batch size 1 for now

        # state = obs["state"].reshape(tuple(obs["state"].shape[:2]) + (-1,))

        # process the point clouds
        # some point clouds might be invalid
        # if this occurs, exclude these batch items
        xyzs = []
        ac = np.zeros([batch_size, self.pred_horizon, self.num_eef, self.dof])

        forward_idxs = list(np.arange(batch_size))
        for batch_idx, xyz in enumerate(obs["pc"]):
            if not batch_idx in forward_idxs:
                xyzs.append(np.zeros((self.num_points, 3)))
            elif xyz.shape[0] == 0:
                # no points in point cloud, return no-op action
                forward_idxs.remove(batch_idx)
                xyzs.append(np.zeros((self.num_points, 3)))
            elif self.shuffle_pc:
                choice = np.random.choice(
                    xyz.shape[0], self.num_points, replace=True
                )
                xyz = xyz[choice, :]
                xyzs.append(xyz)
            else:
                step = xyz.shape[0] // self.num_points
                xyz = xyz[::step, :][: self.num_points]
                xyzs.append(xyz)
        torch_obs = dict(
            pc=torch.tensor(np.array(xyzs)).to(self.device).float(), 
            gt_grasp = obs['gt_grasp']  )
        history = self.actor(torch_obs, return_history=return_history)



            # # unorm history
            # unnormed_history = []
            # if return_history and batch_size == 1:
                
            #     for j in range(len(history)):
            #         grasp_h, joint_h = history[j]
            #         grasp_h = (
            #             self.grasp_xyz_normalizer.unnormalize(grasp_h[i])
            #             .detach()
            #             .cpu()
            #             .numpy()
            #             .reshape(-1)
            #         )
            #         joint_h = (
            #             self.jpose_normalizer.unnormalize(joint_h[i])
            #             .detach()
            #             .cpu()
            #             .numpy()
            #             .reshape(-1)
            #         )
            #         unnormed_history.append((grasp_h, joint_h))


            # ac[idx] = (unnormed_grasp,unnormed_jpose)
        return history
