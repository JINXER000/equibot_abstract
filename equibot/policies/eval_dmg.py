import os
import sys
import time
import torch
import hydra
import omegaconf
import wandb
import numpy as np

from equibot.policies.utils.misc import  EQUIBOT_PATH, get_agent, compose_transformation
from equibot.policies.datasets.robosuite_policy_dataset import RobosuitePolicyDataset

from scripts.robomimic_dmg_wrapper import DMG_env_switchable,to_camel_case, ts_tuple

from typing import Dict, Callable, List

## diffusion_policy/common/pytorch_utils.py
def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result




class Robosuite_Evaluator(DMG_env_switchable):
    def __init__(self, cfg, output, max_timesteps, num_inference_steps, with_planning= False, scale = 1.0, fps = 10, crf = 22, record = False):
        self.cfg = cfg
        self.dataset = RobosuitePolicyDataset(cfg.data.dataset, mode = 'inference')
        self.max_timesteps = max_timesteps
        self.output = output
        self.num_inference_steps = num_inference_steps

        self.with_planning = with_planning  

    def initialize_env(self, cfg, reset_grippers= True, **kwargs):
        self.cur_env_name = cfg.env.env_name
        self.load_checkpoint(cfg, width = 168, height = 168)        
        self.ts = self.reset_all(reset_grippers = reset_grippers)


    def reset_ts(self, with_planning = False):
        self.raw_obs = self.env.reset()
        self.obs = self.collect_obs(self.raw_obs)
        init_ts = ts_tuple(self.obs, 0, False, {})
        return init_ts
    
    def step_ts(self, action):
        self.raw_obs, reward, done, info = self.env.step(action)
        self.obs = self.collect_obs(self.raw_obs)
        info["is_success"] = self.is_success()
        return ts_tuple(self.obs, reward, done, info)
    
    def reset_all(self, reset_grippers = True):
        ## obs history for extracting multi-step obs
        self.obs_history = dict()
        for key in self.obs_shape_meta.keys():
            self.obs_history[key] = np.zeros(
                (self.max_timesteps, *self.obs_shape_meta[key].shape),
                dtype=np.float32
            )
        self.t = 0

        ts = self.reset_ts(with_planning=self.with_planning)
        
        return ts

    def get_seq_obs(self):
        """
        Get a sequence of observations for the policy.
        If we don't have enough history, pad with the first observation.
        """
        obs_dict_np = dict()
        if self.t < self.n_obs_steps - 1:
            # Pad with first observation if we don't have enough history
            for k, v in self.obs_history.items():
                obs_dict_np[k] = np.array([v[0]] * self.n_obs_steps, dtype=np.float32)
                obs_dict_np[k][self.n_obs_steps-self.t-1:self.n_obs_steps] = v[0:self.t+1]
        else:
            # Get the last n_obs_steps observations
            for k, v in self.obs_history.items():
                obs_dict_np[k] = v[self.t-self.n_obs_steps+1:self.t+1]
        return obs_dict_np
    
    def collect_obs(self, obs):

        cur_obs = dict()

        ## gather pc
        pc_dict = self.save_mj_observation(interested_objs= self.related_objects, record_ply=False)
        all_pc = np.concatenate(
            [pc_dict[obj] for obj in self.related_objects], axis=0
        )
        cur_obs['pc'] = self.dataset._downsample_pc(all_pc)
        # import open3d as o3d
        # all_pcd = o3d.geometry.PointCloud()
        # all_pcd.points = o3d.utility.Vector3dVector(cur_obs['pc'])
        # all_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # gray color
        # o3d.io.write_point_cloud(f"pc_{self.t}.ply", all_pcd)

        ## gather proprio
        eef_states = {}
        gripper_states = {}
        for robot_name in self.related_robots:
            biop_eef_pose = obs[f"{robot_name}_eef_pos"]
            biop_eef_quat = obs[f"{robot_name}_eef_quat"]
            eef_state = compose_transformation(biop_eef_pose, biop_eef_quat)
            eef_states[robot_name] = eef_state.reshape(1, 4, 4)

            gripper_state2finger = obs[f"{robot_name}_gripper_qpos"]
            gripper_states[robot_name] = gripper_state2finger[0]

        eef_state_trans = np.concatenate([eef_states[robot_name] for robot_name in self.related_robots], axis=0)
        gripper_vals = np.array([gripper_states[robot_name] for robot_name in self.related_robots]).reshape(-1, 1)

        eef_state_3vec = eef_state_trans[:, :3, [3, 0, 1]].transpose(0, 2, 1)
        eef_state_9d = eef_state_3vec.reshape(self.num_eef, 9)
        gravity_vec = np.array([0, 0, -1])
        gravity_expanded = np.tile(gravity_vec, (self.num_eef, 1))
        eef_state_13d = np.concatenate([eef_state_9d, gravity_expanded, gripper_vals], axis=-1)

        cur_obs['eef_pos'] = eef_state_13d

        ## update obs history
        self.obs_history['pc'][self.t] = cur_obs['pc']
        self.obs_history['eef_pos'][self.t] = eef_state_13d

        return cur_obs 

    def organize_equipolicy_action(self, action):

        action_np = action.detach().to('cpu').numpy()

        action_7d = action_np.reshape(-1, self.num_eef, 7)  # gripper, relpos, relrot
        eef_gripper = action_7d[:,:, 0].reshape(-1, self.num_eef, 1)  # gripper value
        eef_relpos = action_7d[:,:, 1:4]  # 3d position relative to the base
        eef_relaxis = action_7d[:,:, 4:7]  # 3d rotation axis relative to the base
        # TODO: check the code in robomimic inference
        # eef_relrpy[0] = Rotation.from_rotvec(eef_relaxis).as_euler('xyz') # convert to euler angles

        ## re-order for robosuite input format
        # action_7d_dmg = np.concatenate((eef_relpos, np.zeros(eef_relaxis.shape), eef_gripper), axis=-1)  
        action_7d_dmg = np.concatenate((eef_relpos, eef_relaxis, eef_gripper), axis=-1)
        action_out = action_7d_dmg.reshape(-1, 7 * self.num_eef)  

        return action_out
    
    def load_checkpoint(self, cfg, width = 168, height = 168, controller_name = "OSC_POSE", **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load checkpoint
        self.cfg = cfg
        self.agent = get_agent(cfg.agent.agent_name)(cfg)
        self.agent.train(False)
        ckpt_path_full = os.path.join(EQUIBOT_PATH, cfg.training.ckpt)
        self.agent.load_snapshot(ckpt_path_full)

        self.related_objects = self.cfg.data.dataset['related_objects']
        self.related_robots = self.cfg.data.dataset['related_robots']
        self.num_eef = len(self.related_robots)

        # hyper-parameters
        ## observation
        self.obs_shape_meta = cfg.data.shape_meta.obs

        ## multi-step params for policy
        self.query_cycle = cfg.model.pred_horizon
        self.n_obs_steps = cfg.model.obs_horizon

        ## setup environment
        env_name = to_camel_case(self.cur_env_name)

        super().__init__(env_name, controller_name = controller_name, abs_action = False, H = height, W= width, cam_names = ["agentview", "birdview", "frontview", "robot0_eye_in_hand", "robot1_eye_in_hand"],)
        
    def inference_once(self, render = True):
        if self.t >= self.max_timesteps:
            return True
        with torch.inference_mode():
            # process previous ts
            # obs = self.ts.observation
            # self.collect_obs(obs)
            obs_dict_np = self.get_seq_obs()
            agent_obs = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

            # query policy to extract action: (B=1, Da)
            # t0 = time.perf_counter()
            if self.t % self.query_cycle == 0:
                # action_dict = self.policy.predict_action(obs_dict)
                predicted_action = self.agent.predict_action(agent_obs)
                self.np_action_seq = self.organize_equipolicy_action(predicted_action)
            total_action = self.np_action_seq[self.t % self.query_cycle]
            # t1 = time.perf_counter()
            # total_action = self.organize_equipolicy_action(agent_ac)
            self.ts = self.step_ts(total_action)

            self.t += 1

        if render:
            self.env.render()

        # self.record_frame(obs)

        return self.ts.done
    
def wrapper_test(cfg):

    max_timesteps = 500
    num_inference_steps = 10

    output = './outputs/robosuite_eval'
    env_runer = Robosuite_Evaluator(cfg, output, max_timesteps, num_inference_steps =num_inference_steps)
    
    env_runer.initialize_env(cfg)
    for i in range(max_timesteps):
        done = env_runer.inference_once()
        task_success = env_runer.handle_rewards()
        if task_success:
            print('Task completed!')
            break
        # dp.append_image()

    env_runer.exit(output)

@hydra.main(config_path="configs", config_name="dmg_assemly_policy")
def main(cfg):
    cfg.mode == "inference"
    wrapper_test(cfg)

    # device = torch.device(cfg.device)

    # agent = get_agent(cfg.agent.agent_name)(cfg)
    # agent.train(False)
    # ckpt_full_path = os.path.join(EQUIBOT_PATH, cfg.training.ckpt)
    # agent.load_snapshot(ckpt_full_path)

    # dmg_dir = cfg.eval.dmg_dir
    # sys.path.append(dmg_dir)
    # from scripts.robomimic_dmg_wrapper import DMG_env_switchable

    # task_name = 'two_arm_three_piece_assembly'
    # env_runner = DMG_env_switchable(task_name)
    # dataset_path = cfg.data.dataset.path
    # obs = env_runner.rollout_from_biop(dataset_path)

    # env_runner.get_equipolicy_agent(agent, equi_cfg=cfg)

    # if cfg.use_wandb:
    #     wandb_config = omegaconf.OmegaConf.to_container(
    #         cfg, resolve=True, throw_on_missing=False
    #     )
    #     wandb.init(
    #         entity=cfg.wandb.entity,
    #         project=cfg.wandb.project,
    #         tags=["eval"],
    #         name=cfg.prefix,
    #         settings=wandb.Settings(code_dir="."),
    #         config=wandb_config,
    #     )
    # np.random.seed(cfg.seed)


if __name__ == "__main__":
    main()