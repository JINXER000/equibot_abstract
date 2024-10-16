import numpy as np
import torch
from torch import nn

from equibot.policies.utils.norm import Normalizer
from equibot.policies.utils.misc import to_torch
from equibot.policies.utils.diffusion.lr_scheduler import get_scheduler

from equibot.policies.agents.compaloha_policy import CompALOHAPolicy

class CompALOHAAgent(object):
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
        # self.num_eef = cfg.env.num_eef
        self.num_eef = self.actor.num_eef
        self.dof = cfg.env.dof
        self.num_points = cfg.data.dataset.num_points
        self.obs_mode = cfg.model.obs_mode
        self.ac_mode = cfg.model.ac_mode
        self.obs_horizon = cfg.model.obs_horizon
        self.pred_horizon = cfg.model.pred_horizon
        self.shuffle_pc = cfg.data.dataset.shuffle_pc

        self.all_normalizers = None

        self.symb_mask = cfg.data.dataset.symb_mask


    def _init_actor(self):
        self.actor = CompALOHAPolicy(self.cfg, device=self.cfg.device).to(self.cfg.device)
        self.actor.ema.averaged_model.to(self.cfg.device)

    def get_jpose_normalizer(self, joint_data):
        flattend_joint_data = joint_data.view(-1, self.dof)
        indices = [[i for i in range(self.dof)]]
        jpose_normalizer = Normalizer(flattend_joint_data, symmetric=True, indices=indices)
        return jpose_normalizer

    
    def get_xyz_normalizer(self, xyz_data):
        flattend_xyz = xyz_data.view(-1, 3)
        indices = [[0,1,2]]
        xyz_normalizer = Normalizer(flattend_xyz, symmetric=True, indices=indices)
        return xyz_normalizer
    
    def get_pc_scale(self, pc_data, ac_scale):
        pc = pc_data.reshape(-1, self.num_points, 3)
        centroid = pc.mean(1, keepdim=True)
        centered_pc = pc - centroid
        pc_scale = centered_pc.norm(dim=-1).mean()
        # ac_scale = pc_normalizer.stats["max"].max()
        normed_pc_scale = pc_scale / ac_scale
        return normed_pc_scale


    def _init_multi_normalizers(self, batch_dict):
        all_normalizers = {}
        for side in ['left', 'right']:
            jpose_normalizer = self.get_jpose_normalizer(batch_dict[side+'_jpose'])
            pc_normalizer = self.get_xyz_normalizer(batch_dict[side+'_pc'])
            grasp_normalizer = Normalizer(
                {
                    "min": pc_normalizer.stats["min"],
                    "max": pc_normalizer.stats["max"],
                }
            )
            all_normalizers[side+'_jpose'] = jpose_normalizer
            all_normalizers[side+'_pc'] = pc_normalizer
            all_normalizers[side +'_grasp'] = grasp_normalizer
            
            all_normalizers[side+'_pc_scale'] = self.get_pc_scale(batch_dict[side+'_pc'], pc_normalizer.stats["max"].max())
        return all_normalizers
    
    
    
    def train(self, training=True):
        self.actor.nets.train(training)



    def update(self, batch, vis=False):
        self.train()

        ###### Load data, preprocessing using mask ######
        batch = to_torch(batch, self.device)
        left_pc = batch['left_pc'].repeat(1, self.obs_horizon, 1, 1)
        right_pc = batch['right_pc'].repeat(1, self.obs_horizon, 1, 1)
        left_jpose = batch['left_jpose'].repeat(1, self.pred_horizon, 1, 1)
        right_jpose = batch['right_jpose'].repeat(1, self.pred_horizon, 1, 1)
        left_grasp = batch['left_grasp'].repeat(1, self.pred_horizon, 1, 1)
        right_grasp = batch['right_grasp'].repeat(1, self.pred_horizon, 1, 1)


        ### init normalizers
        n_data_dict = {
            'left_pc': left_pc,
            'right_pc': right_pc,
            'left_grasp': left_grasp,
            'right_grasp': right_grasp,
            'left_jpose': left_jpose,
            'right_jpose': right_jpose,
        }
        if self.all_normalizers is None:
            self.all_normalizers = self._init_multi_normalizers(n_data_dict)
            self.actor.all_normalizers = self.all_normalizers

        ### proc pc
        left_obs_vec, left_center, left_scale = self.actor.proc_pc(left_pc, 'left_pc')
        right_obs_vec, right_center, right_scale = self.actor.proc_pc(right_pc, 'right_pc')


        # proc grasp pose. first split flattened pose to xyz and dir, then normalize xyz. 
        gt_left_grasp_z = self.actor.proc_grasp(left_grasp, 'left_grasp', left_center, left_scale)
        gt_right_grasp_z = self.actor.proc_grasp(right_grasp, 'right_grasp', right_center, right_scale)

        # scalar
        scalar_left_jpose = self.actor.proc_jpose(left_jpose, 'left_jpose')
        scalar_right_jpose = self.actor.proc_jpose(right_jpose, 'right_jpose')
        
        batch_size = left_pc.shape[0]
        timesteps = torch.randint(
            0,
            self.actor.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()

        ######## train the pred net ########

        metrics = {
                   "left_obsv": np.linalg.norm(
                       left_obs_vec.detach().cpu().numpy(), axis=1
                   ).mean(),
                     "right_obsv": np.linalg.norm(
                          right_obs_vec.detach().cpu().numpy(), axis=1
                     ).mean(),
                   }
        
        ## z_t
        left_jpose_noise = torch.randn_like(scalar_left_jpose, device=self.device)
        ## x_t = add_noise(x_0, z_t)
        noisy_left_jpose = self.actor.noise_scheduler.add_noise(
            scalar_left_jpose, left_jpose_noise, timesteps
        )
        right_jpose_noise = torch.randn_like(scalar_right_jpose, device=self.device)
        noisy_right_jpose = self.actor.noise_scheduler.add_noise(
            right_jpose_noise, scalar_right_jpose, timesteps
        )

        left_grasp_noise = torch.randn_like(gt_left_grasp_z, device=self.device)
        noisy_left_grasp = self.actor.noise_scheduler.add_noise(
            gt_left_grasp_z, left_grasp_noise, timesteps
        )

        right_grasp_noise = torch.randn_like(gt_right_grasp_z, device=self.device)
        noisy_right_grasp = self.actor.noise_scheduler.add_noise(
            gt_right_grasp_z, right_grasp_noise, timesteps
        )


        ## /tilde{z}_t = prednet(x_t, Cond, t)
        left_vec_noise_pred, left_scalar_noise_pred = self.actor.left_noise_pred_net_handle(
            noisy_left_grasp,
            timesteps,
            scalar_sample = noisy_left_jpose, 
            cond=left_obs_vec,
            scalar_cond=None,
        )
        left_vec_loss = nn.functional.mse_loss(left_vec_noise_pred, gt_left_grasp_z)
        metrics["left_vec_loss"] = left_vec_loss
        left_scalar_loss = nn.functional.mse_loss(left_scalar_noise_pred, left_jpose_noise)
        metrics["left_scalar_loss"] = left_scalar_loss

        ## debug: only use left network
        total_loss = left_scalar_loss + left_vec_loss
        metrics["log_loss"] = np.log(total_loss.detach().cpu().numpy())


        # right_vec_noise_pred, right_scalar_noise_pred = self.actor.right_noise_pred_net_handle(
        #     noisy_right_grasp,
        #     timesteps,
        #     scalar_sample = noisy_right_jpose, 
        #     cond=right_obs_vec,
        #     scalar_cond=None,
        # )
        # right_vec_loss = nn.functional.mse_loss(right_vec_noise_pred, gt_right_grasp_z)
        # metrics["right_vec_loss"] = right_vec_loss
        # right_scalar_loss = nn.functional.mse_loss(right_scalar_noise_pred, right_jpose_noise)
        # metrics["right_scalar_loss"] = right_scalar_loss

        # total_loss = left_scalar_loss + right_scalar_loss + left_vec_loss + right_vec_loss
        # metrics["log_loss"] = np.log(total_loss.detach().cpu().numpy())

        if torch.isnan(total_loss):
            print(f"Loss is nan, please investigate.")
            import pdb

            pdb.set_trace()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.actor.step_ema()

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
        )

        for side in ["left", "right"]:
            state_dict[f"{side}_pc_scale"] = self.all_normalizers[f"{side}_pc_scale"]
            state_dict[f"{side}_pc_normalizer"] = self.all_normalizers[f"{side}_pc"].state_dict()
            state_dict[f"{side}_grasp_normalizer"] = self.all_normalizers[f"{side}_grasp"].state_dict()
            state_dict[f"{side}_jpose_normalizer"] = self.all_normalizers[f"{side}_jpose"].state_dict()

        torch.save(state_dict, save_path)

    def load_snapshot(self, load_path):
        state_dict = torch.load(load_path)
        self.all_normalizers = {}
        for side in ["left", "right"]:
            self.all_normalizers[f"{side}_pc_scale"] = state_dict[f"{side}_pc_scale"]
            self.all_normalizers[f"{side}_pc"] = Normalizer(state_dict[f"{side}_pc_normalizer"])
            self.all_normalizers[f"{side}_grasp"] = Normalizer(state_dict[f"{side}_grasp_normalizer"])
            self.all_normalizers[f"{side}_jpose"] = Normalizer(state_dict[f"{side}_jpose_normalizer"])
        
        self.actor.all_normalizers = self.all_normalizers


        del self.actor.left_encoder_handle
        del self.actor.right_encoder_handle
        del self.actor.left_noise_pred_net_handle
        del self.actor.right_noise_pred_net_handle    
        self.actor.load_state_dict(self.fix_checkpoint_keys(state_dict["actor"]))
        self.actor._init_torch_compile()

        self.actor.ema.averaged_model.load_state_dict(
            self.fix_checkpoint_keys(state_dict["ema_model"])
        )

        

    ## call this function during evaluation (only during training)
    def act(self, batch, history_bid = -1):
        self.train(False)


        # batch_size = obs["left_pc"].shape[0]
        # # batch_size = 1  # only support batch size 1 for now
        # assert history_bid < batch_size # batch to select as denoising history

        # assert obs['left_pc'].shape[2] == self.num_points
        # assert obs['right_pc'].shape[2] == self.num_points

        # torch_obs = dict(
        #     left_pc=torch.tensor(obs['left_pc']).to(self.device).float(), 
        #     right_pc=torch.tensor(obs['right_pc']).to(self.device).float(),
        #     left_grasp = torch.tensor(obs['left_grasp']).to(self.device).float(),
        #     right_grasp = torch.tensor(obs['right_grasp']).to(self.device).float(),
        #     left_jpose = torch.tensor(obs['left_jpose']).to(self.device).float(),
        #     right_jpose = torch.tensor(obs['right_jpose']).to(self.device).float(),
        #     )
        action_dict, eval_metrics, denoise_history = self.actor(batch, history_bid=history_bid)


        return denoise_history, eval_metrics
