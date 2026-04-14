import copy
import hydra
from omegaconf import OmegaConf
import torch
from torch import nn
import numpy as np

from equibot.policies.vision.sim3_encoder import SIM3Vec4Latent
from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.equivariant_diffusion.conditional_unet1d import VecConditionalUnet1D, FeatFusion
from equibot.policies.utils.equivariant_diffusion.unconditional_mlp import UnconditionalMLP
from equibot.policies.utils.normalizer import LinearNormalizer

from equibot.policies.utils.lan_utils import get_and_save_skill_bert_embs, MLPEncoder

from equibot.policies.utils.misc import to_torch, \
    convert_trans_to_vec, convert_vec_to_trans, ActionSlice,\
     geodestDist, EQUIBOT_PATH, to_torch, to_tensor,\
    convert_trans_to_4pts, convert_4pts_to_trans, matrix_to_rotation_6d, render_trajectory, ascii_tensor_batch_to_str, vis_metric_imgs

    
class EquiSkillPolicy(nn.Module):
    def __init__(self, cfg,  device="cpu"):
        nn.Module.__init__(self)
        self.cfg = cfg
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

        self.normalizer = LinearNormalizer()
        self.statistics = {}

        if hasattr(cfg.model, "num_diffusion_iters"):
            self.num_diffusion_iters = cfg.model.num_diffusion_iters
        else:
            self.num_diffusion_iters = cfg.model.noise_scheduler.num_train_timesteps


        self.encoder_out_dim = cfg.model.encoder.c_dim

        # self.separate_policy = cfg.model.separate_policy

        self.dof = cfg.env.dof # 6
        self.num_eef = cfg.env.num_eef

        self.obs_dim = self.encoder_out_dim

        # Get use_pc_color flag from config (check both model and dataset configs)
        self.use_pc_color = cfg.model.get('use_pc_color', False) or cfg.data.dataset.get('use_pc_color', False)

        net_dict = {}
        # Pass use_rgb to encoder backbone args if using color
        # Safely copy encoder config (OmegaConf can be in struct mode)
        encoder_cfg = OmegaConf.to_container(cfg.model.encoder, resolve=True)
        encoder_cfg = copy.deepcopy(encoder_cfg)
        backbone_type = encoder_cfg.get('backbone_type', 'vn_pointnet')
        
        # VDGCNN backbone does not support RGB (preloaded weights don't expect it)
        if self.use_pc_color and "VDGCNN" in backbone_type:
            print(f"[EquiSkillPolicy] Warning: use_pc_color=True but backbone_type={backbone_type} does not support RGB. Disabling RGB.")
            self.use_pc_color = False
        
        backbone_args = copy.deepcopy(encoder_cfg.get('backbone_args', {}))
        if self.use_pc_color:
            backbone_args['use_rgb'] = True
        encoder_cfg['backbone_args'] = backbone_args
        net_dict['obj_encoder'] = SIM3Vec4Latent(**encoder_cfg)

        # self.eef_dims = {}

        ##  set up language encoder TODO: check if it is saved in ckpt
        language_encoder_cfg = cfg.model.language_encoder_cfg
        output_size = language_encoder_cfg.hidden_size
        assert output_size == self.encoder_out_dim
        self.language_encoder = self._setup_language_encoder(output_size=output_size, **language_encoder_cfg)
        ## will be included into ema_model further
        net_dict['language_encoder'] = self.language_encoder 

        # self.skill_names = None ## to be loaded

        ## eef_representation can be vectors or points
        self.eef_representation = cfg.data.dataset.eef_representation
        if self.eef_representation == "3vec":
            self.eef_proc_fn = self.proc_eef_3vec
            self.eef_recover_fn = self.recover_eef_3vec
            self.eef_dims = 3
        elif self.eef_representation == "4pts":
            self.eef_proc_fn = self.proc_eef_4pts
            self.eef_recover_fn = self.recover_eef_4pts
            ## expand to B, H, 4, 3
            self.original_gripper_pcd = np.array(cfg.data.dataset.original_gripper_pcd)
            self.eef_dims = 4
        else:
            raise ValueError(f"Unsupported eef_representation: {self.eef_representation}")

        policy_key = 'unitraj_noise_pred_net'

        # scalar_cond_dim = self.encoder_out_dim * self.obs_horizon
        # skill_names + task_names
        scalar_cond_dim = self.encoder_out_dim * self.obs_horizon * 2

        net_dict[policy_key] = VecConditionalUnet1D(
            input_dim=self.eef_dims,  ## vec dim, rot is 2, xyz is 1
            cond_dim=self.obs_dim* self.obs_horizon,
            scalar_cond_dim= scalar_cond_dim,  ## if =1,  it is the skill_emb_batch
            scalar_input_dim= 1,  ## output gripper val
            diffusion_step_embed_dim=self.obs_dim* self.obs_horizon,
            cond_predict_scale=False,
            down_dims=cfg.model.down_dims,
            )

        ## check if fuse_inv_feat in cfg
        try:
            self.fuse_inv_feat = cfg.model.fuse_inv_feat
        except:
            self.fuse_inv_feat = False

        if self.fuse_inv_feat:
            ## input and output are equiv feat, cond is inv feat
            net_dict['feat_fusion'] = FeatFusion(
                input_dim=self.obs_dim* self.obs_horizon,
                output_dim=self.obs_dim* self.obs_horizon,
                scalar_cond_dim= self.obs_dim* self.obs_horizon,
            )
            
        self.nets = nn.ModuleDict(net_dict)

        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        self._init_torch_compile()

        self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler)
        
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized DMG Policy with {num_parameters} parameters")


    def _init_torch_compile(self):
        nets_handles = {}
        if self.use_torch_compile:
            for key, net in self.nets.items():
                nets_handles[key] = torch.compile(net)
        else:
            nets_handles = self.nets

        self.nets = nets_handles



    def _setup_language_encoder(self, network_name, **language_encoder_kwargs):
        return eval(network_name)(**language_encoder_kwargs)
    
    def get_encoding_from_name_batch(self, name_batch, batch_size, mapping_dict):

        if batch_size == 1:
            emb_tensor = mapping_dict[name_batch]
            if isinstance(emb_tensor, np.ndarray):
                emb_tensor = torch.tensor(emb_tensor).to(self.device)
            emb_batch = self.encode_bert_emb(emb_tensor, batch_size)
        else:
            emb_tensor_batch = []
            for skill_name in name_batch:
                emb_tensor = mapping_dict[skill_name]
                if isinstance(emb_tensor, np.ndarray):
                    emb_tensor = torch.tensor(emb_tensor).to(self.device)
                emb_tensor_batch.append(emb_tensor)
            emb_tensor_batch = torch.stack(emb_tensor_batch, dim=0)
            emb_batch = self.encode_bert_emb(emb_tensor_batch, batch_size)
        return emb_batch

    def encode_bert_emb(self, bert_emb, batch_size):
        skill_emb = self.nets['language_encoder'](bert_emb)
        skill_emb_batch = skill_emb.reshape(batch_size, -1)       
        return skill_emb_batch

    def step_ema(self):
        self.ema.step(self.nets)

    def normalize_from_key(self, key, data):
        if self.cfg.data.dataset.normalization_method == "batch":
            return self.all_normalizers[key].normalize(data)
        else:
            return self.normalizer[key].normalize(data)
    
    def unnormalize_from_key(self, key, data):
        if self.cfg.data.dataset.normalization_method == "batch":
            return self.all_normalizers[key].unnormalize(data)
        else:
            return self.normalizer[key].unnormalize(data)

    def recover_eef_3vec(self, eefpos_batch, scale, center, key):
        side = key.split('_')[0]

        # reshape dim to B,  (3 or 6) , 3
        # eefpos_batch = torch.mean(eefpos_batch, dim=1, keepdim=True)
        scale = torch.mean(scale, dim=2, keepdim=True)
        center = torch.mean(center, dim=2, keepdim=True)

        batch_size = eefpos_batch.shape[0]

        ##### eefpos processing
        eefpos_xyz = eefpos_batch[:, :,0, :].reshape(batch_size, -1, 1, 3)

        # add back the offset
        eefpos_xyz = eefpos_xyz *scale + center

        # un-normalize
        unnormed_eefpos_xyz = (
                    self.unnormalize_from_key(key, eefpos_xyz)
                )
        
        ##### rotation processing 
        rot6d_batch = eefpos_batch[:, :, 1: , :].reshape(batch_size, -1, 1, 6)

        trans_batch = convert_vec_to_trans(rot6d_batch, unnormed_eefpos_xyz)

        # trans_batch = trans_batch.detach().cpu().numpy()

        return trans_batch, unnormed_eefpos_xyz, rot6d_batch
    
    ## eefpt_batch is B, H, 4, 3
    def recover_eef_4pts(self, eefpt_batch, scale, center, key):
        scale = torch.mean(scale, dim=2, keepdim=True)
        center = torch.mean(center, dim=2, keepdim=True)
        
        eefpt_batch_uncano = eefpt_batch * scale + center
        unnormed_eefpt_batch = self.unnormalize_from_key(key, eefpt_batch_uncano)

        trans_batch = convert_4pts_to_trans(unnormed_eefpt_batch, self.original_gripper_pcd)
        
        ## only for eval
        unnormed_eefpos_xyz = trans_batch[:, :, :3, 3:].transpose(-2, -1)
        eef_rot = trans_batch[:, :, :3, :3].reshape(-1, 3, 3)
        rot6d_batch = matrix_to_rotation_6d(eef_rot)

        # trans_batch = trans_batch.detach().cpu().numpy()
        return trans_batch, unnormed_eefpos_xyz, rot6d_batch



    def recover_gripper(self, gripper_batch, key):
        unnormed_gripper = (
                    self.unnormalize_from_key(key, gripper_batch)
                    .detach()
                    .cpu()
                    .numpy()
                )
        return unnormed_gripper

    def proc_pc(self, pc,  ema_nets = None):
        pc_key = 'pc'
        pc = self.normalize_from_key(pc_key, pc)
        batch_size = pc.shape[0]

        ## Extract RGB from point cloud if using color
        if self.use_pc_color and pc.shape[-1] >= 6:
            pc, rgb = pc[..., :3], pc[..., 3:6]
        else:
            rgb = None

        ## in training
        encoder_key = 'obj_encoder'
        if ema_nets is None:
            encoder_handle = self.nets[encoder_key] 
        else:
            encoder_handle = ema_nets[encoder_key]
        pc_scale = self.statistics['pc_scale']

        feat_dict = encoder_handle(pc, target_norm=pc_scale, rgb=rgb)

        center = (
            feat_dict["center"].reshape(batch_size, self.obs_horizon, 1, 3)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        )
        scale = feat_dict["scale"].reshape(batch_size, self.obs_horizon, 1, 1)[:, [-1]].repeat(1, self.pred_horizon, 1, 1)
        equiv_feat = feat_dict["so3"]  
        equiv_feat = equiv_feat.reshape(batch_size, -1, 3)
        # inv_feat = feat_dict["inv"].reshape(batch_size, -1)
        return equiv_feat,  center, scale

    def get_in_hand_inv_feat(self, in_hand_pc, in_hand_kw,  ema_nets = None):
        # pc_key = 'in_hand_pc'
        in_hand_pc = self.normalize_from_key(in_hand_kw, in_hand_pc)
        batch_size = in_hand_pc.shape[0]

        ## Extract RGB from point cloud if using color
        if self.use_pc_color and in_hand_pc.shape[-1] >= 6:
            in_hand_pc, rgb = in_hand_pc[..., :3], in_hand_pc[..., 3:6]
        else:
            rgb = None

        ## in training
        encoder_key = 'obj_encoder'
        if ema_nets is None:
            encoder_handle = self.nets[encoder_key] 
        else:
            encoder_handle = ema_nets[encoder_key]

        if f'{in_hand_kw}_scale' in self.statistics:
            pc_scale = self.statistics[f'{in_hand_kw}_scale'] 
        else:
            pc_scale = self.statistics['pc_scale']

        feat_dict = encoder_handle(in_hand_pc, target_norm=pc_scale, rgb=rgb)

        inv_feat = feat_dict["inv"].reshape(batch_size, -1)

        return inv_feat
    
    # def revise_inv_feat_using_mask(self, in_hand_pc, in_hand_mask, inv_feat, ema_nets = None):
    #     in_hand_pc_data = in_hand_pc.repeat(1, self.obs_horizon, 1, 1)
    #     in_hand_inv_feat = self.get_in_hand_inv_feat(in_hand_pc_data, ema_nets = ema_nets)

    #     B, L = in_hand_inv_feat.shape

    #     in_hand_mask = in_hand_mask.to(device=in_hand_pc.device, dtype=torch.bool)
    #     expanded_mask = in_hand_mask.view(B, 1).expand(B, L)
    #     inv_feat = torch.where(expanded_mask, in_hand_inv_feat , inv_feat)
    #     return inv_feat
    
    def combine_inv_feat_and_so3_feat(self, inv_feat, so3_feat):
        # batch_size = so3_feat.shape[0]
        # inv_feat = inv_feat.reshape(batch_size, -1, 1)
        # return inv_feat * so3_feat
        return self.nets['feat_fusion'](so3_feat, inv_feat)

    # in dataset, first pc is converted using min(). Then, in pc_normalizer, pc.max is mapped to 1. in eef normalizer, eef_xyz = traj = (traj-pc.min)/pc.max.  Here center should be 0.5, and scale be 1. finally, eef_xyz mean shoule be near 0. 
    def proc_eef_3vec(self, eef_pose, key, center, scale):
        eef_xyz_raw, eef_dir1, eef_dir2 = convert_trans_to_vec(eef_pose)
        eef_xyz = self.normalize_from_key(key, eef_xyz_raw)
        eef_xyz = (eef_xyz - center)/scale
        gt_eef_z = torch.cat([eef_xyz, eef_dir1, eef_dir2], dim=-2)

        return gt_eef_z
    
    ## TODO: check if the output ranges from -1 to 1
    def proc_eef_4pts(self, eef_pose, key, center, scale):
        eef_4pts_raw = convert_trans_to_4pts(eef_pose, self.original_gripper_pcd)
        eef_4pts = self.normalize_from_key(key, eef_4pts_raw)
        eef_4pts = (eef_4pts - center) / scale


        return eef_4pts

    def proc_gripper(self, raw_gripper, key):
        gripper_action = self.normalize_from_key(key, raw_gripper)
        gripper_action = gripper_action.reshape(gripper_action.shape[0], -1, 1)
        return gripper_action


    def get_all_embs(self, skill_name_batch, batch_size, task_name_batch = None):
        skill_emb_batch = self.get_encoding_from_name_batch(skill_name_batch, batch_size, self.statistics['skill_embs_all_tasks'])

        if task_name_batch is not None:
            task_emb_batch = self.get_encoding_from_name_batch(task_name_batch, batch_size, self.statistics['task_emb_dict'])
            skill_emb_batch = torch.cat([skill_emb_batch, task_emb_batch], dim=-1)
        return skill_emb_batch
    
    def fuse_in_hand_pc(self, agent_obs, obs_vec, should_fuse, ema_nets = None):
        if not should_fuse:
            return obs_vec
        
        in_hand_kws = [kw for kw in agent_obs.keys() if 'in_hand_pc' in kw]
        in_hand_kws = sorted(in_hand_kws)
        ## debug: use explicit list
        # in_hand_kws = ['left_in_hand_pc', 'right_in_hand_pc']
        for in_hand_kw in in_hand_kws:
            in_hand_pc_data = agent_obs[in_hand_kw].repeat(1, self.obs_horizon, 1, 1)
            inv_feat = self.get_in_hand_inv_feat(in_hand_pc_data, in_hand_kw,  ema_nets = ema_nets)
            obs_vec = self.combine_inv_feat_and_so3_feat(inv_feat, obs_vec)

        return obs_vec

    def pred_unimaual_traj(self, skill_name_batch, agent_obs, gt_batch = None, task_name_batch = None):
        pc_data = agent_obs['pc'].repeat(1, self.obs_horizon, 1, 1)
        batch_size =  pc_data.shape[0]

        ema_nets = self.ema.averaged_model

        obs_vec,  center, scale = self.proc_pc(pc_data, ema_nets = ema_nets)

        obs_vec = self.fuse_in_hand_pc(agent_obs, obs_vec, self.fuse_inv_feat, ema_nets = ema_nets)

        ##### start denoising #####
        initial_noise_scale = 1
        noisy_eef_xt = torch.randn((batch_size, self.pred_horizon, self.eef_dims, 3)).to(self.device)\
        * initial_noise_scale

        noisy_gripper = torch.randn((batch_size, self.pred_horizon, 1)).to(self.device) * initial_noise_scale

        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        curr_action = { "eefpos": noisy_eef_xt, 
            "gripper": noisy_gripper}
        
         ####### inverse diffusion step
        policy_key = 'unitraj_noise_pred_net'
        task_skill_condition = self.get_all_embs(skill_name_batch, batch_size, task_name_batch)
        
        # obs_vec = equiv_feat
        # task_skill_condition = torch.cat([inv_feat, task_skill_condition], dim=-1)

        for k in self.noise_scheduler.timesteps:

            new_action = { "eefpos": None, "gripper": None }

            vec_noise_pred, gripper_noise_pred = ema_nets[policy_key](\
                sample=curr_action["eefpos"],
                timestep = k,
                scalar_sample = curr_action["gripper"], 
                cond= obs_vec,
                scalar_cond=task_skill_condition,
            )
            new_action["eefpos"] = self.noise_scheduler.step(
                model_output=vec_noise_pred, timestep=k, sample=curr_action["eefpos"]
            ).prev_sample

            new_action["gripper"] = self.noise_scheduler.step(
                model_output=gripper_noise_pred, timestep=k, sample=curr_action["gripper"]
            ).prev_sample

            curr_action = new_action

        ### recover the eefpos eef pose
        ## predicted values
        trans_batch, unnormed_eefpos_xyz, rot6d_batch = self.eef_recover_fn(\
            curr_action["eefpos"], scale, center, key="eefpos")
        assert trans_batch.shape[3] == 4

        ## recover gripper
        gripper_batch = self.recover_gripper(curr_action["gripper"], key="gripper")

        ## update action dict 
        action_dict = {}
        eval_metrics = {}
        if batch_size ==1:
            action_dict["eefpos"] = trans_batch[0]
            action_dict["gripper"] = gripper_batch[0]
        ## calc metrics if in training
        else:
            if self.eef_representation == "4pts":
                gt_4pts =  self.eef_proc_fn(gt_batch["eefpos"], 'eefpos', center, scale)
                pred_4pts = curr_action["eefpos"]
                pts_error = torch.nn.functional.mse_loss(pred_4pts, gt_4pts)
                eval_metrics["pts_error"] = pts_error

            pred_xyz = trans_batch[:, :, :3, 3]
            gt_xyz = gt_batch["eefpos"][:, :, :3, 3]
            xyz_l1 = torch.nn.functional.l1_loss(pred_xyz, gt_xyz)
            eval_metrics["xyz_l1"] = xyz_l1

            gt_Rs = gt_batch["eefpos"][:, :, :3, :3]
            pred_Rs = trans_batch[:, :, :3, :3]
            diff_theta = geodestDist(gt_Rs, pred_Rs).mean()
            eval_metrics["rot_diff"] = diff_theta * 180 / torch.pi

            plotted_titles = []
            for i in range(batch_size):
                skill_name = skill_name_batch[i]
                if task_name_batch is not None:
                    task_name = task_name_batch[i]
                else:
                    task_name = ''
                title = f'{skill_name}-{task_name}-prediction'
                if title not in plotted_titles:
                    plotted_titles.append(title)
                else:
                    continue

                pc_data = agent_obs['pc'][i,0].detach().cpu().numpy()  # Shape: (N, 3)
                trajectory = trans_batch[i].detach().cpu().numpy()  # Shape: (T, 4, 4)
                gripper_values = gripper_batch[i]  # Shape: (T,)
                rendered_img = render_trajectory(pc_data, trajectory, gripper_values, title = title)
            
                # Store the rendered image in eval_metrics
                eval_metrics[f"{title}-image"] = rendered_img

        return action_dict, eval_metrics


    def skill_task_ascii_to_str(self, batch):
        skill_name_batch = ascii_tensor_batch_to_str(batch['skill_name'])
        task_name_batch = ascii_tensor_batch_to_str(batch['task_name'])
        return skill_name_batch, task_name_batch

    def forward(self, batch, skill_id=-1):
        ###### preprocess data from dataset #######
        batch = to_torch(batch, self.device)

        action_dict_all = {}
        eval_metrics_all = {}

        # self.load_skill_name_to_emb()
        assert self.skill_names is not None
        assert self.task_names is not None

        skill_name_batch, task_name_batch = self.skill_task_ascii_to_str(batch)

        agent_obs = {}
        for key, value in batch.items():
            if 'pc' in key:
                agent_obs[key] = value

        # pc_data = batch['pc']
        # agent_obs = {'pc': pc_data}
        # if 'in_hand_pc' in batch:
        #     agent_obs['in_hand_pc'] = batch['in_hand_pc']
            # agent_obs['in_hand_mask'] = batch['in_hand_mask']
        action_dict, eval_metrics = self.pred_unimaual_traj(skill_name_batch, agent_obs, gt_batch=batch, task_name_batch=task_name_batch)
        action_dict_all.update(action_dict)
        eval_metrics_all.update(eval_metrics)

        denoise_history = []
        if skill_id >= 0 and len(action_dict_all) > 0:
            ## in traj mode, we do not visulize the history. Instead, we visualize the final action
            skill_name = self.skill_names[skill_id]
            traj_len = self.pred_horizon
            for i in range(traj_len):
                action_slice = ActionSlice(mode="separated")

                action_slice.update(f"{skill_name}:eefpos", action_dict_all[f"{skill_name}:eefpos"][i])
                action_slice.update(f"{skill_name}:gripper", action_dict_all[f"{skill_name}:gripper"][i])

                denoise_history.append(action_slice)
        return action_dict_all, eval_metrics_all, denoise_history

class BiopSkillPolicy(nn.Module):
    def __init__(self, cfg,  device="cpu"):
        nn.Module.__init__(self)
        self.cfg = cfg
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

        self.normalizer = LinearNormalizer()
        self.statistics = {}

        if hasattr(cfg.model, "num_diffusion_iters"):
            self.num_diffusion_iters = cfg.model.num_diffusion_iters
        else:
            self.num_diffusion_iters = cfg.model.noise_scheduler.num_train_timesteps


        self.encoder_out_dim = cfg.model.encoder.c_dim

        # self.separate_policy = cfg.model.separate_policy

        self.dof = cfg.env.dof # 6
        self.num_eef = cfg.env.num_eef

        self.obs_dim = self.encoder_out_dim

        net_dict = {}
        # net_dict['obj_encoder'] = SIM3Vec4Latent(**cfg.model.encoder)

        # self.eef_dims = {}

        ##  set up language encoder TODO: check if it is saved in ckpt
        language_encoder_cfg = cfg.model.language_encoder_cfg
        output_size = language_encoder_cfg.hidden_size
        assert output_size == self.encoder_out_dim
        self.language_encoder = self._setup_language_encoder(output_size=output_size, **language_encoder_cfg)
        ## will be included into ema_model further
        net_dict['language_encoder'] = self.language_encoder 


        joint_scalar_dims = self.dof * self.num_eef  
        
        ### Get unconditional MLP configuration
        if hasattr(cfg.model, 'unconditional_mlp_cfg'):
            mlp_cfg = cfg.model.unconditional_mlp_cfg
        else:
            mlp_cfg = None
            
        diffusion_step_embed_dim = self.obs_dim * self.obs_horizon

        net_dict['jpose_noise_pred_net'] = UnconditionalMLP(
            input_dim=joint_scalar_dims,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            cfg=mlp_cfg
        )

        self.nets = nn.ModuleDict(net_dict)

        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        self._init_torch_compile()

        self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler)
        
        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized DMG Policy with {num_parameters} parameters")


    def _init_torch_compile(self):
        nets_handles = {}
        if self.use_torch_compile:
            for key, net in self.nets.items():
                nets_handles[key] = torch.compile(net)
        else:
            nets_handles = self.nets

        self.nets = nets_handles

    def proc_jpose(self, jpose, key):
        
        jpose_n = self.normalize_from_key(key, jpose)
        # jpose_vec = self._convert_jpose_to_vec(jpose_n)
        jpose_vec = jpose_n.reshape(jpose_n.shape[0], -1,  self.dof * self.num_eef)
        return jpose_vec
    

    def recover_jpose(self, jpose_batch, key):
        # squeeze the pred horizon to 1
        jpose_action = jpose_batch.reshape(-1,  self.num_eef, self.dof)
        unnormed_joint = (
                    self.unnormalize_from_key(key, jpose_action)
                    .detach()
                    .cpu()
                    .numpy()
                )
        
        return unnormed_joint

    def _setup_language_encoder(self, network_name, **language_encoder_kwargs):
        return eval(network_name)(**language_encoder_kwargs)
    
    def get_encoding_from_name_batch(self, name_batch, batch_size, mapping_dict):

        if batch_size == 1:
            emb_tensor = mapping_dict[name_batch]
            if isinstance(emb_tensor, np.ndarray):
                emb_tensor = torch.tensor(emb_tensor).to(self.device)
            emb_batch = self.encode_bert_emb(emb_tensor, batch_size)
        else:
            emb_tensor_batch = []
            for skill_name in name_batch:
                emb_tensor = mapping_dict[skill_name]
                if isinstance(emb_tensor, np.ndarray):
                    emb_tensor = torch.tensor(emb_tensor).to(self.device)
                emb_tensor_batch.append(emb_tensor)
            emb_tensor_batch = torch.stack(emb_tensor_batch, dim=0)
            emb_batch = self.encode_bert_emb(emb_tensor_batch, batch_size)
        return emb_batch

    def encode_bert_emb(self, bert_emb, batch_size):
        skill_emb = self.nets['language_encoder'](bert_emb)
        skill_emb_batch = skill_emb.reshape(batch_size, -1)       
        return skill_emb_batch

    def step_ema(self):
        self.ema.step(self.nets)

    def normalize_from_key(self, key, data):
        if self.cfg.data.dataset.normalization_method == "batch":
            return self.all_normalizers[key].normalize(data)
        else:
            return self.normalizer[key].normalize(data)
    
    def unnormalize_from_key(self, key, data):
        if self.cfg.data.dataset.normalization_method == "batch":
            return self.all_normalizers[key].unnormalize(data)
        else:
            return self.normalizer[key].unnormalize(data)



    def get_all_embs(self, skill_name_batch, batch_size, task_name_batch = None):
        skill_emb_batch = self.get_encoding_from_name_batch(skill_name_batch, batch_size, self.statistics['skill_embs_all_tasks'])

        if task_name_batch is not None:
            task_emb_batch = self.get_encoding_from_name_batch(task_name_batch, batch_size, self.statistics['task_emb_dict'])
            skill_emb_batch = torch.cat([skill_emb_batch, task_emb_batch], dim=-1)
        return skill_emb_batch

    
    def pred_bimanual_jposes(self, skill_name_batch, agent_obs, gt_batch = None, task_name_batch = None):
        if isinstance(skill_name_batch, str):
            batch_size = 1
        else:
            batch_size = len(skill_name_batch)
            
        ema_nets = self.ema.averaged_model

        initial_noise_scale = 1
        noisy_jpose = torch.randn((batch_size,   self.num_eef*self.dof)).to(self.device) * initial_noise_scale

        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        curr_action = {'jpose': noisy_jpose}

        # ## TODO: condition on skill name and task name
        # emb_batch = self.get_all_embs(skill_name_batch, batch_size, task_name_batch)

        ####### inverse diffusion step
        for k in self.noise_scheduler.timesteps:
            biop_key = 'jpose'
            new_action = {biop_key: None}

            scalar_noise_pred = ema_nets['jpose_noise_pred_net'](\
                sample=curr_action[biop_key],
                timesteps = k,
            )
            new_action[biop_key] = self.noise_scheduler.step(
                model_output=scalar_noise_pred, timestep=k, sample=curr_action[biop_key]
            ).prev_sample

            curr_action = new_action

        unnormed_joint = self.recover_jpose(curr_action[biop_key], key=biop_key)
        unnormed_joint = torch.tensor(unnormed_joint).to(self.device)   

        action_dict = {}
        eval_metrics = {}
        if batch_size ==1:
            action_dict[biop_key] = unnormed_joint.reshape(self.num_eef, self.dof)
        else:
            gt_joint = gt_batch['jpose'].reshape(-1, self.num_eef, self.dof)
            joint_mse = torch.nn.functional.mse_loss(unnormed_joint, gt_joint)
            eval_metrics["dual_joint_mse"] = joint_mse

        return action_dict, eval_metrics
    

    def skill_task_ascii_to_str(self, batch):
        skill_name_batch = ascii_tensor_batch_to_str(batch['skill_name'])
        task_name_batch = ascii_tensor_batch_to_str(batch['task_name'])
        return skill_name_batch, task_name_batch

    def forward(self, batch, skill_id=-1):
        ###### preprocess data from dataset #######
        batch = to_torch(batch, self.device)

        action_dict_all = {}
        eval_metrics_all = {}

        # self.load_skill_name_to_emb()
        assert self.skill_names is not None
        assert self.task_names is not None

        skill_name_batch, task_name_batch = self.skill_task_ascii_to_str(batch)

        agent_obs = None
        action_dict, eval_metrics = self.pred_bimanual_jposes(skill_name_batch, agent_obs, gt_batch=batch, task_name_batch=task_name_batch)
        action_dict_all.update(action_dict)
        eval_metrics_all.update(eval_metrics)


        return action_dict_all, eval_metrics_all, None

 