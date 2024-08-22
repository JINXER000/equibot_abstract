import copy
import hydra
import torch
from torch import nn
import torch.nn.functional as F

from equibot.policies.vision.sim3_encoder import SIM3Vec4Latent
from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.equivariant_diffusion.conditional_unet1d import VecConditionalUnet1D
import numpy as np

from aloha_policy import ALOHAPolicy


class AnnealedULASampler:
    """ Implements AIS with ULA """

    def __init__(self,
                 num_samples_per_step,
                 step_sizes,
                 gradient_function,
                 noise_function):
        self._step_sizes = step_sizes
        if isinstance(num_samples_per_step, int):
            num_samples_per_step = torch.tensor([num_samples_per_step] * len(step_sizes))
        self._num_samples_per_step = num_samples_per_step
        self._gradient_function = gradient_function
        self._noise_function = noise_function

    @torch.enable_grad()
    def sample_step(self, x, batch, t):
        if self._num_samples_per_step.device != t.device:
            self._num_samples_per_step = self._num_samples_per_step.to(t.device)
        for i in range(self._num_samples_per_step[t]):
            ss = self._step_sizes[t]
            std = (2 * ss) ** .5
            grad = self._gradient_function(x, batch, t)
            noise = self._noise_function() * std
            x = x + grad * ss + noise

        return x
    
    
class CompALOHAPolicy(ALOHAPolicy):
    def __init__(self, cfg, device="cpu"):
        nn.Module.__init__(self)
        self.obs_mode = cfg.model.obs_mode
        self.ac_mode = cfg.model.ac_mode
        self.use_torch_compile = cfg.model.use_torch_compile
        self.device = device

        # |o|o|                             observations: 2
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        # | |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        self.pred_horizon = cfg.model.pred_horizon
        self.obs_horizon = cfg.model.obs_horizon
        self.action_horizon = cfg.model.ac_horizon
        self.symb_mask = cfg.data.dataset.symb_mask

        if hasattr(cfg.model, "num_diffusion_iters"):
            self.num_diffusion_iters = cfg.model.num_diffusion_iters
        else:
            self.num_diffusion_iters = cfg.model.noise_scheduler.num_train_timesteps

        if self.obs_mode.startswith("pc"):
            self.encoder = SIM3Vec4Latent(**cfg.model.encoder) # hidden_dim = 32
        else:
            self.encoder = None
        self.encoder_out_dim = cfg.model.encoder.c_dim

        # self.num_eef = cfg.env.num_eef
        self.dof = cfg.env.dof # 6
        self.eef_dim = 3 # xyz, dir1, dir2
        self.num_eef = len([x for x in self.symb_mask[:2] if x != 'None'])
        num_scalar_dims = self.dof * self.num_eef # joint pose

        self.obs_dim = self.encoder_out_dim
        # grasp pose
        self.grasp_noise_pred_net = VecConditionalUnet1D(
            input_dim=self.eef_dim,
            cond_dim=self.obs_dim* self.obs_horizon,
            scalar_cond_dim=0,
            scalar_input_dim= 0,
            diffusion_step_embed_dim=self.obs_dim* self.obs_horizon,
            cond_predict_scale=False,
        )
        # joint pose
        self.jpose_noise_pred_net = VecConditionalUnet1D(
            input_dim=self.eef_dim,
            cond_dim=0,
            scalar_cond_dim=0,
            scalar_input_dim= num_scalar_dims,
            diffusion_step_embed_dim=self.obs_dim* self.obs_horizon,
            cond_predict_scale=True,
        )

        self.nets = nn.ModuleDict(
            {"encoder": self.encoder, "grasp_noise_pred_net": self.grasp_noise_pred_net,
             "jpose_noise_pred_net": self.jpose_noise_pred_net}
        )
        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        self._init_torch_compile()

        self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler)

        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized paraGen Policy with {num_parameters} parameters")

    def _init_torch_compile(self):
        if self.use_torch_compile:
            self.encoder_handle = torch.compile(self.encoder)
            self.grasp_noise_pred_net_handle = torch.compile(self.grasp_noise_pred_net)
            self.jpose_noise_pred_net_handle = torch.compile(self.jpose_noise_pred_net)
        else:
            self.encoder_handle = self.encoder
            self.grasp_noise_pred_net_handle = self.grasp_noise_pred_net
            self.jpose_noise_pred_net_handle = self.jpose_noise_pred_net


    
    def forward(self, obs, history_bid=-1):
        ###### preprocess data #######
        pc = obs["pc"]
        pc = pc.repeat(1, self.obs_horizon, 1, 1)
        pc = self.pc_normalizer.normalize(pc)
        
        batch_size =  pc.shape[0]

        ema_nets = self.ema.averaged_model
        feat_dict = ema_nets["encoder"](pc, ret_perpoint_feat=True, target_norm=self.pc_scale)
        center = (
            feat_dict["center"].reshape(batch_size, self.obs_horizon, 1, 3)[:, [-1]].repeat(1, self.obs_horizon, 1, 1)
        )
        scale = (
            feat_dict["scale"].reshape(batch_size, self.obs_horizon, 1, 1)[:, [-1]].repeat(1, self.obs_horizon, 1, 1)
        )

        pc_feat = feat_dict["so3"]  
        obs_cond_vec = pc_feat.reshape(batch_size, -1, 3)

        if self.mask_type == 'both':
            #### define EBM ####
            def gradient_function(x, batch, t):
                gradient = - self.denoise_fn(x, batch, t, eval=True) \
                            * self._sqrt_recipm1_alphas_cumprod_custom[t]
                # print('gradient_function', x.shape, gradient.shape)
                return gradient


            def noise_function():
                return torch.randn(shape, device=device)
            
            # ULA sampler
            samples_per_step = self.samples_per_step
            step_sizes = self.step_sizes
            sampler = AnnealedULASampler(samples_per_step, step_sizes, gradient_function, noise_function)

        ##### start denoising #####

        initial_noise_scale = 1
        noisy_grasp = (
            torch.randn((batch_size, self.pred_horizon, self.eef_dim, 3)).to(self.device)
            * initial_noise_scale,
            None,
        )
        noisy_jpose = (
            torch.randn((batch_size, self.pred_horizon, self.num_eef*self.dof)).to(self.device)
            * initial_noise_scale,
        )

        curr_action = [noisy_grasp, noisy_jpose]
        self.noise_scheduler.set_timestamps(self.num_diffusion_iters)

        denoise_history = []
        for k in self.noise_scheduler.timesteps:

            ####### inverse diffusion step
            new_action = [None, None]

            if self.mask_type != 'only_jpose':            
                # load from existing data statistics
                # predict noise
                grasp_noise_pred = ema_nets["grasp_noise_pred_net"](
                    sample=curr_action[0],
                    timestep=k,
                    scalar_sample=None,
                    cond=obs_cond_vec,
                )


                new_action[0] = self.noise_scheduler.step(
                    model_output=grasp_noise_pred, timestep=k, sample=curr_action[0]
                ).prev_sample

            if self.mask_type != 'only_grasp':

                jpose_noise_pred = ema_nets["jpose_noise_pred_net"](
                    sample=torch.randn_like(curr_action[0]),
                    timestep=k,
                    scalar_sample=curr_action[1],
                    cond=None,
                )

                new_action[1] = self.noise_scheduler.step(
                    model_output=jpose_noise_pred, timestep=k, sample=curr_action[1]
                ).prev_sample
                   
            if self.mask_type == 'both':
                # do EBM
                if k % self.ebm_per_steps == 0:
                    pose_features = sampler.sample_step(pose_features, batch, t)
                    # print(f'p_sample_loop {j}/{self.num_timesteps}')

            # record history
            if history_bid >=0:
                if self.mask_type != 'only_jpose':
                    trans_batch, _, _ = self.recover_grasp(new_action[0], scale, center)
                    assert trans_batch.shape[3] == 4
                    trans_mat = trans_batch[history_bid, 0].detach().cpu().numpy()
                
                if self.mask_type != 'only_grasp':
                    unnormed_joint = self.recover_jpose(noise_pred[1])
                    jpose_flat = unnormed_joint[history_bid].reshape(-1)
                    jpose_12d = np.zeros(12)
                    if self.symb_mask[0] == 'None':  # only right
                        jpose_12d[6:] = jpose_flat
                    elif self.symb_mask[1] == 'None':
                        jpose_12d[:6] = jpose_flat
                    else:
                        jpose_12d = jpose_flat
                    action_slice = (trans_mat, jpose_12d)
                else:
                    action_slice = (trans_mat, None)
                denoise_history.append(action_slice)

            # record the denoised action
            curr_action = tuple(new_action)
