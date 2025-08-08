import numpy as np
import torch
from torch import nn

from equibot.policies.utils.norm import Normalizer
from equibot.policies.utils.misc import to_torch, \
    ascii_tensor_to_str
from equibot.policies.utils.diffusion.lr_scheduler import get_scheduler

from equibot.policies.agents.per_skill_policy import EquiSkillPolicy
from equibot.policies.utils.misc import to_torch,  rotate_observation, to_tensor, EQUIBOT_PATH , ascii_tensor_to_str



class EquiSkillAgent(object):
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

        # self.symb_mask = cfg.data.dataset.symb_mask


    def _init_actor(self):
        self.actor = EquiSkillPolicy(self.cfg, device=self.cfg.device).to(self.cfg.device)
        self.actor.ema.averaged_model.to(self.cfg.device)

    

    
    def get_pc_scale(self, pc_data, ac_scale):
        pc = pc_data.reshape(-1, self.num_points, 3)
        centroid = pc.mean(1, keepdim=True)
        centered_pc = pc - centroid
        pc_scale = centered_pc.norm(dim=-1).mean()
        normed_pc_scale = pc_scale / ac_scale
        return normed_pc_scale

    def get_xyz_normalizer(self, xyz_data):
        flattend_xyz = xyz_data.view(-1, 3)
        indices = [[0,1,2]]
        xyz_normalizer = Normalizer(flattend_xyz, symmetric=True, indices=indices)
        return xyz_normalizer

    def _init_multi_normalizers(self, n_data_dict):
        all_normalizers = {}

            ## pc normalizer
        pc_normalizer = self.get_xyz_normalizer(n_data_dict['pc'])
        all_normalizers['pc'] = pc_normalizer
        self.actor.statistics['pc_scale'] = self.get_pc_scale(n_data_dict['pc'], pc_normalizer.stats["max"].max())

        ## skill normalizer
        grasp_normalizer = Normalizer(
            {
                "min": pc_normalizer.stats["min"],
                "max": pc_normalizer.stats["max"],
            }
        )
        all_normalizers['eefpos'] = grasp_normalizer

        gripper_normalizer = Normalizer(n_data_dict['gripper'], symmetric=True, indices=[[0]])
        all_normalizers['gripper'] = gripper_normalizer

        return all_normalizers
    ## max: 0.1598; min: 0
    
    def train(self, training=True):
        self.actor.nets.train(training)



    def learn_unimanual_traj(self,  n_data_dict):
        ## cached pc feature
        obs_vec, center, scale = self.actor.proc_pc(n_data_dict['pc'])

        eefpos = n_data_dict['eefpos']
        gripper = n_data_dict['gripper']
        bert_emb = n_data_dict['skill_name_emb']

        ## proc grasp
        gt_eefpos_z = self.actor.eef_proc_fn(eefpos, 'eefpos', center, scale)

        ## proc gripper
        gt_gripper_action = self.actor.proc_gripper(gripper, 'gripper')

        batch_size = eefpos.shape[0]
        timesteps = torch.randint(
            0,
            self.actor.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        
        ## z_t
        ## x_t = add_noise(x_0, z_t)

        eefpos_noise = torch.randn_like(gt_eefpos_z, device=self.device)
        noisy_eefpos = self.actor.noise_scheduler.add_noise(gt_eefpos_z, eefpos_noise, timesteps)

        gripper_action_noise = torch.randn_like(gt_gripper_action, device=self.device)
        noisy_gripper_action = self.actor.noise_scheduler.add_noise(gt_gripper_action, gripper_action_noise, timesteps)

        ## /tilde{z}_t = prednet(x_t, Cond, t)
        policy_key = 'unitraj_noise_pred_net'
        skill_scalar_id = self.actor.encode_bert_emb(bert_emb, batch_size)

        eefpos_noise_pred, gripper_noise_pred = self.actor.nets[policy_key](
            noisy_eefpos,
            timesteps,
            scalar_sample = noisy_gripper_action,
            cond = obs_vec,
            scalar_cond = skill_scalar_id,
        )
        
        vec_loss = nn.functional.mse_loss(eefpos_noise_pred, eefpos_noise)
        scalar_loss = nn.functional.mse_loss(gripper_noise_pred, gripper_action_noise)

        return vec_loss, scalar_loss

    ## NOTE: this function only for libero single robot skill update
            ## as skill_type is different in a batch, it is hard to branching
    def update(self, batch):
        self.train()

        ###### Load data, preprocessing using mask ######
        batch = to_torch(batch, self.device)
        n_data_dict = {}
        n_data_dict['skill_name_emb'] = batch['skill_name_emb']
        n_data_dict['pc'] = batch['pc']
        n_data_dict['eefpos'] = batch['eefpos']
        n_data_dict['gripper'] = batch['gripper']
        # n_data_dict['skill_name'] = ascii_tensor_to_str(batch['skill_name'])

        n_data_dict['pc'] = batch['pc'].repeat(1, self.obs_horizon, 1, 1)


        ## assume the data is already normalized
        if self.all_normalizers is None and self.cfg.data.dataset.normalization_method == "batch":
            self.all_normalizers = self._init_multi_normalizers(n_data_dict)
            self.actor.all_normalizers = self.all_normalizers


    ######## train the pred net ########
        metrics = {}
        vec_loss, scalar_loss = self.learn_unimanual_traj(n_data_dict)
        metrics['vec_loss'] = vec_loss
        metrics['scalar_loss'] = scalar_loss

        total_loss = 0
        for metric_key in metrics:
            if metric_key.endswith('_loss'):
                total_loss += metrics[metric_key]

        metrics["log_loss"] = np.log(total_loss.detach().cpu().numpy())

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

    def set_normalizer_and_statistics(self, dataset):
        if self.cfg.data.dataset.normalization_method == "all":
            self.set_normalizer(dataset.normalizer.state_dict())
            self.actor.statistics = dataset.statistics

    def set_normalizer(self, normalizer_state_dict):
        self.actor.normalizer.load_state_dict(normalizer_state_dict)


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
        if self.cfg.data.dataset.normalization_method == "all":
            state_dict["normalizer"] = self.actor.normalizer.state_dict()
            state_dict["statistics"] = self.actor.statistics
        else:

            state_dict["eefpos_normalizer"] = self.all_normalizers["eefpos"].state_dict()
            state_dict["gripper_normalizer"] = self.all_normalizers["gripper"].state_dict()
            state_dict["pc_scale"] = self.actor.statistics["pc_scale"]
            state_dict["pc_normalizer"] = self.all_normalizers["pc"].state_dict()

        torch.save(state_dict, save_path)

    ## TODO: check if the normalizer is loaded correctly
    def load_snapshot(self, load_path):
        import os
        load_path_full = os.path.join(EQUIBOT_PATH, load_path)
        state_dict = torch.load(load_path_full)
    
        self.actor.load_state_dict(self.fix_checkpoint_keys(state_dict["actor"]))
        self.actor._init_torch_compile()

        self.actor.ema.averaged_model.load_state_dict(
            self.fix_checkpoint_keys(state_dict["ema_model"])
        )

        if self.cfg.data.dataset.normalization_method == "all":
            self.set_normalizer(state_dict["normalizer"])
            self.actor.statistics = state_dict["statistics"]
        else:

            self.all_normalizers = {}
            self.all_normalizers["eefpos"] =Normalizer(state_dict["eefpos_normalizer"])
            self.all_normalizers["gripper"] = Normalizer(state_dict["gripper_normalizer"])
            self.all_normalizers["pc"] = Normalizer(state_dict["pc_normalizer"])
            self.actor.all_normalizers = self.all_normalizers

            self.actor.statistics["pc_scale"] = state_dict["pc_scale"]


    ## call this function during evaluation (only during training)
    def eval_with_rotation(self, obs, skill_id = -1):
        self.train(False)
        random_yaw = np.random.uniform(-np.pi, np.pi)
        np_obs= rotate_observation(obs, random_yaw)
        cpu_obs = to_tensor(np_obs)
        gpu_obs = to_torch(cpu_obs, self.device)

        # gpu_obs = obs
        with torch.no_grad():
            action_dict, eval_metrics, denoise_history = self.actor(gpu_obs, skill_id=skill_id)

        return denoise_history, eval_metrics
