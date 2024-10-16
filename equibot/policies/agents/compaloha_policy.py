import copy
import hydra
import torch
from torch import nn
import torch.nn.functional as F

from equibot.policies.vision.sim3_encoder import SIM3Vec4Latent
from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.equivariant_diffusion.conditional_unet1d import VecConditionalUnet1D
import numpy as np

from equibot.policies.utils.misc import to_torch, \
    convert_trans_to_vec, convert_vec_to_trans, ActionSlice


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
    
    
class CompALOHAPolicy(nn.Module):
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
        self.has_eff = ('predeff' in cfg.data.dataset.dataset_type)

        if hasattr(cfg.model, "num_diffusion_iters"):
            self.num_diffusion_iters = cfg.model.num_diffusion_iters
        else:
            self.num_diffusion_iters = cfg.model.noise_scheduler.num_train_timesteps

        self.left_encoder = SIM3Vec4Latent(**cfg.model.encoder) # hidden_dim = 32
        self.right_encoder = SIM3Vec4Latent(**cfg.model.encoder) # hidden_dim = 32

        self.encoder_out_dim = cfg.model.encoder.c_dim

        # self.num_eef = cfg.env.num_eef
        self.dof = cfg.env.dof # 6
        self.eef_dim = 3 # xyz, dir1, dir2
        self.num_eef = cfg.env.num_eef
        num_scalar_dims = self.dof * self.num_eef # joint pose

        self.obs_dim = self.encoder_out_dim

        self.left_noise_pred_net = VecConditionalUnet1D(
            input_dim=self.eef_dim,
            cond_dim=self.obs_dim* self.obs_horizon,
            scalar_cond_dim=0,
            scalar_input_dim= num_scalar_dims,
            diffusion_step_embed_dim=self.obs_dim* self.obs_horizon,
            cond_predict_scale=True,  ## in Fila, do AX+B instead of x+B
        )
        self.right_noise_pred_net = VecConditionalUnet1D(
            input_dim=self.eef_dim,
            cond_dim=self.obs_dim* self.obs_horizon,
            scalar_cond_dim=0,
            scalar_input_dim= num_scalar_dims,
            diffusion_step_embed_dim=self.obs_dim* self.obs_horizon,
            cond_predict_scale=True,
        )

        self.nets = nn.ModuleDict(
            {"left_encoder": self.left_encoder, \
                "right_encoder": self.right_encoder, \
             "left_noise_pred_net": self.left_noise_pred_net,\
             "right_noise_pred_net": self.right_noise_pred_net}
        )
        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        self._init_torch_compile()

        self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler)

        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized paraGen Policy with {num_parameters} parameters")

    def _init_torch_compile(self):
        if self.use_torch_compile:
            self.left_encoder_handle = torch.compile(self.left_encoder)
            self.right_encoder_handle = torch.compile(self.right_encoder)
            self.left_noise_pred_net_handle = torch.compile(self.left_noise_pred_net)
            self.right_noise_pred_net_handle = torch.compile(self.right_noise_pred_net)
        else:
            self.left_encoder_handle = self.left_encoder
            self.right_encoder_handle = self.right_encoder
            self.left_noise_pred_net_handle = self.left_noise_pred_net
            self.right_noise_pred_net_handle = self.right_noise_pred_net

    # def _convert_jpose_to_vec(self, jpose, batch=None):
    #     # input: (B, 1, E , dof); output: (B, 1, ac_dim, 3) 
    #     # jpose = jpose.reshape(jpose.shape[0], jpose.shape[1],  -1, 3)
    #     jpose = jpose.reshape(jpose.shape[0], -1,  self.dof * self.num_eef)
    #     return jpose
    
    def step_ema(self):
        self.ema.step(self.nets)

    def normalize_from_key(self, key, data):
        return self.all_normalizers[key].normalize(data)
    
    def unnormalize_from_key(self, key, data):
        return self.all_normalizers[key].unnormalize(data)

    def recover_grasp(self, grasp_batch, scale, center, key):
        # reshape dim to B,  (3 or 6) , 3
        grasp_batch = torch.mean(grasp_batch, dim=1, keepdim=True)
        scale = torch.mean(scale, dim=1, keepdim=True)
        center = torch.mean(center, dim=1, keepdim=True)

        ##### grasp processing
        if self.has_eff == False:
            grasp_xyz = grasp_batch[:, :,0, :].reshape(-1, 1, 1, 3)
        else:
            grasp_xyz = grasp_batch[:, :, :2, :].reshape(-1, 1, 2, 3)

        # add back the offset
        grasp_xyz = grasp_xyz *scale + center

        # un-normalize
        unnormed_grasp_xyz = (
                    self.unnormalize_from_key(key, grasp_xyz)
                )
        
        ##### rotation processing 
        if self.has_eff == False:
            rot6d_batch = grasp_batch[:, :, 1: , :].reshape(-1, 1, 1, 6)
        else:
            # rot6d_batch = grasp_batch[:, :, 2:, :].reshape(-1, 1, 2, 6)
            grasp_dir1 = grasp_batch[:, :, 2:4, :]
            grasp_dir2 = grasp_batch[:, :, 4:6, :]
            rot6d_batch = torch.cat((grasp_dir1, grasp_dir2), dim=-1)
            assert rot6d_batch.shape[-1] == 6

        trans_batch = convert_vec_to_trans(rot6d_batch, unnormed_grasp_xyz, has_eff=self.has_eff)

        trans_batch = trans_batch.detach().cpu().numpy()

        return trans_batch, unnormed_grasp_xyz, rot6d_batch

    def recover_jpose(self, jpose_batch, key):
        # reduce the horizon
        jpose_action = torch.mean(jpose_batch, dim=1)
        jpose_action = jpose_action.reshape(-1, self.num_eef, self.dof)
    

        unnormed_joint = (
                    self.unnormalize_from_key(key, jpose_action)
                    .detach()
                    .cpu()
                    .numpy()
                )
        
        return unnormed_joint

    def proc_pc(self, pc, key, ema_nets = None):
        pc = self.normalize_from_key(key, pc)
        batch_size = pc.shape[0]
        side = key.split('_')[0]
        ## in training
        if ema_nets is None:
            encoder_handle = self.left_encoder if side == 'left' else self.right_encoder
            feat_dict = encoder_handle(pc, target_norm=self.all_normalizers[key+'_scale'])
        else: # in inference
            feat_dict = ema_nets[side+"_encoder"](pc, ret_perpoint_feat=True, target_norm=self.all_normalizers[key+'_scale'])
        
        center = (
            feat_dict["center"].reshape(batch_size, self.obs_horizon, 1, 3)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        )
        scale = feat_dict["scale"].reshape(batch_size, self.obs_horizon, 1, 1)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        pc_feat = feat_dict["so3"]  
        obs_cond_vec = pc_feat.reshape(batch_size, -1, 3)
        return obs_cond_vec, center, scale

    def proc_grasp(self, grasp_pose, key, center, scale):
        grasp_xyz_raw, grasp_dir1, grasp_dir2 = convert_trans_to_vec(grasp_pose, has_eff=self.has_eff)
        grasp_xyz = self.normalize_from_key(key, grasp_xyz_raw)
        grasp_xyz = (grasp_xyz - center)/scale
        gt_grasp_z = torch.cat([grasp_xyz, grasp_dir1, grasp_dir2], dim=-2)

        # ## check same
        # trans_batch, unnormed_grasp_xyz, rot6d_batch = self.recover_grasp(gt_grasp_z, scale, center, key)
        # trans_batch_ts = torch.tensor(trans_batch).to(self.device)
        # error = nn.functional.mse_loss(trans_batch_ts, grasp_pose)
        # assert error < 1e-6
        return gt_grasp_z
    
    def proc_jpose(self, jpose, key):
        
        jpose_n = self.normalize_from_key(key, jpose)
        # jpose_vec = self._convert_jpose_to_vec(jpose_n)
        jpose_vec = jpose_n.reshape(jpose_n.shape[0], -1,  self.dof * self.num_eef)
        return jpose_vec
    
    ## TODO: revise it for 2 pc
    def forward(self, batch, history_bid=-1):
        ###### preprocess data #######
        batch = to_torch(batch, self.device)
        left_pc = batch['left_pc'].repeat(1, self.obs_horizon, 1, 1)
        right_pc = batch['right_pc'].repeat(1, self.obs_horizon, 1, 1)
        
        batch_size =  left_pc.shape[0]

        ema_nets = self.ema.averaged_model

        left_obs_vec, left_center, left_scale = self.proc_pc(left_pc, 'left_pc', ema_nets = ema_nets)
        right_obs_vec, right_center, right_scale = self.proc_pc(right_pc, 'right_pc', ema_nets = ema_nets)

        obs_vec = {"left": left_obs_vec, "right": right_obs_vec}
        center = {"left": left_center, "right": right_center}
        scale = {"left": left_scale, "right": right_scale}

        # #### define EBM ####
        # def gradient_function(x, batch, t):
        #     gradient = - self.denoise_fn(x, batch, t, eval=True) \
        #                 * self._sqrt_recipm1_alphas_cumprod_custom[t]
        #     # print('gradient_function', x.shape, gradient.shape)
        #     return gradient


        # def noise_function():
        #     return torch.randn(shape, device=device)
        
        # # ULA sampler
        # samples_per_step = self.samples_per_step
        # step_sizes = self.step_sizes
        # sampler = AnnealedULASampler(samples_per_step, step_sizes, gradient_function, noise_function)

        ##### start denoising #####

        initial_noise_scale = 1
        noisy_left_xt = (
            torch.randn((batch_size, self.pred_horizon, self.eef_dim, 3)).to(self.device)
        * initial_noise_scale,
            torch.randn((batch_size, self.pred_horizon, self.num_eef*self.dof)).to(self.device)
            * initial_noise_scale,
            )
        noisy_right_xt = (
            torch.randn((batch_size, self.pred_horizon, self.eef_dim, 3)).to(self.device)
        * initial_noise_scale,
            torch.randn((batch_size, self.pred_horizon, self.num_eef*self.dof)).to(self.device)
            * initial_noise_scale,
        )

        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        curr_action = {"left": noisy_left_xt, "right": noisy_right_xt}

        denoise_history = []
        for k in self.noise_scheduler.timesteps:

            ####### inverse diffusion step
            new_action = {"left": [None, None], "right": [None, None]}

            for side in ["left", "right"]:
                vec_noise_pred, scalar_noise_pred = ema_nets[side+"_noise_pred_net"](\
                    sample=curr_action[side][0],
                    timestep = k,
                    scalar_sample = curr_action[side][1],
                    cond= obs_vec[side],
                    scalar_cond=None,
                )
                new_action[side][0] = self.noise_scheduler.step(
                    model_output=vec_noise_pred, timestep=k, sample=curr_action[side][0]
                ).prev_sample
                new_action[side][1] = self.noise_scheduler.step(
                    model_output=scalar_noise_pred, timestep=k, sample=curr_action[side][1]
                ).prev_sample

            

            # if self.mask_type == 'both':
            #     # do EBM
            #     if k % self.ebm_per_steps == 0:
            #         pose_features = sampler.sample_step(pose_features, batch, t)
            #         # print(f'p_sample_loop {j}/{self.num_timesteps}')

            # record history in inference (not train)
            if history_bid >=0:
                action_slice = ActionSlice(mode="separated")

                for side in ["left", "right"]:

                    ## recover grasp
                    trans_batch, _, _ = self.recover_grasp(\
                        new_action[side][0], scale[side], center[side], key=side+'_grasp')
                    assert trans_batch.shape[3] == 4
                    trans_mat = trans_batch[history_bid, 0]#.detach().cpu().numpy() 
                    action_slice.update(side+'_grasp', trans_mat)

                    ## recover jpose
                    unnormed_joint = self.recover_jpose(new_action[side][1], key=side+'_jpose')
                    jpose_flat = unnormed_joint[history_bid].reshape(-1)       
                    action_slice.update(side+'_jpose', jpose_flat)           

                denoise_history.append(action_slice)

            # record the denoised action
            curr_action = new_action

        action_dict, eval_metrics = self.get_action_dict_and_metrics(batch, curr_action, center, scale)
        return action_dict, eval_metrics, denoise_history

    def get_action_dict_and_metrics(self, batch, final_action, center, scale):
        eval_metrics = {}
        action_dict = {}
        for side in ["left", "right"]:
            ## predicted values
            trans_batch, unnormed_grasp_xyz, rot6d_batch = self.recover_grasp(\
                final_action[side][0], scale[side], center[side], key=side+'_grasp')
            unnormed_joint = self.recover_jpose(final_action[side][1], key=side+'_jpose')
            unnormed_joint = torch.tensor(unnormed_joint).to(self.device)   

            ## update action dict if only test
            batch_size = trans_batch.shape[0] 
            if batch_size == 1:
                action_dict[side+'grasp'] = trans_batch.reshape(-1, 4)
                action_dict[side+'jpose'] = unnormed_joint.reshape(self.num_eef, self.dof)

            ## calc metrics if in training
            else:
                gt_grasp_xyz, gt_dir1, gt_dir2 = convert_trans_to_vec(batch[side+"_grasp"], has_eff=self.has_eff)
                # gt_grasp_xyz = torch.mean(gt_grasp_xyz, dim=1, keepdim=True)
                gt_grasp_rot6d = torch.cat([gt_dir1, gt_dir2], dim=-1)
                # gt_grasp_rot6d = torch.mean(gt_grasp_rot6d, dim=1, keepdim=True)
                gt_joint = batch[side+"_jpose"]

                xyz_mse = torch.nn.functional.mse_loss(unnormed_grasp_xyz, gt_grasp_xyz)
                rot_mse = torch.nn.functional.mse_loss(rot6d_batch, gt_grasp_rot6d)
                joint_mse = torch.nn.functional.mse_loss(unnormed_joint, gt_joint)

                eval_metrics[side+"_xyz_mse"] = xyz_mse
                eval_metrics[side+"_rot_mse"] = rot_mse
                eval_metrics[side+"_joint_mse"] = joint_mse

        return action_dict, eval_metrics

    def conclude_masks(self):
        has_grasp = False
        has_jpose = False
        if self.symb_mask[0] != 'None' or self.symb_mask[1] != 'None':
            has_jpose = True
        if self.symb_mask[2] != 'None' or self.symb_mask[3] != 'None':
            has_grasp = True
        if has_grasp and has_jpose:
            return "both"
        elif has_grasp:
            return "only_grasp"
        elif has_jpose:
            return "only_jpose"