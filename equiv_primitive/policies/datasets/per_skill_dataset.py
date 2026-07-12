import os
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from equiv_primitive.policies.utils.misc import (
    EQUIV_PRIMITIVE_PATH,
    compose_transformation,
    centralize_downsample,
    add_pcd_noise,
    centralize_grasp,
    choose_ids,
    choose_ids_rdp,
    rotate_dataslice,
    get_rbt_states,
    get_rbt_actions,
    get_pc_instances,
    get_sg,
    convert_trans_to_vec,
    convert_trans_to_4pts,
    str_to_ascii_tensor,
    get_obj_visibility,
)

from equiv_primitive.policies.utils.lan_utils import get_embs_without_saving, save_embs

from equiv_primitive.policies.utils.normalize_utils import to_torch_stats, get_torch_range_symmetric_normalizer_from_stat, get_torch_isotropic_xyz_normalizer_from_stat

from equiv_primitive.policies.utils.normalizer import LinearNormalizer

## substring marking a bimanual skill name when splitting data slices
BIMANUAL_KW = 'bimanual'

## default mapping from scene-graph neighbor to point-cloud slot; override via cfg.nbr_side_mapping
DEFAULT_NBR_SIDE_MAPPING = {'robot0': 'left_in_hand_pc', 'robot1': 'right_in_hand_pc', 'table': 'pc'}

## default number of frames searched for a visible object pc; override via cfg.visibility_search_window
DEFAULT_VISIBILITY_SEARCH_WINDOW = 8


def get_random_in_hand_pc(ref_pc):
    ## input is (1, num_points, 3)
    with torch.no_grad():
        mean = ref_pc.mean(dim=1, keepdim=True)  # (1, 1, 3)
        std = ref_pc.std(dim=1, keepdim=True, unbiased=False)  # (1, 1, 3)
        eps = 1e-6
        std = torch.clamp(std, min=eps)
        rand_norm = torch.randn_like(ref_pc)
        in_hand_obj_pc_tensor = rand_norm * std + mean
    return in_hand_obj_pc_tensor


class PerSkillDataset(Dataset):
    def __init__(self, cfg, mode, transform=None, pre_transform=None, pre_filter=None, **kwargs):
        super().__init__()
        self.mode = mode
        self.dir_name = cfg.path
        self.root = self.dir_name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.composed_inference = False

        self.use_pc_color = cfg.get('use_pc_color', False)
        # Update pc_shape based on whether color is used
        pc_channels = 6 if self.use_pc_color else 3
        self.pc_shape = (cfg.num_points, pc_channels)

        self.is_obj_centric = cfg.is_obj_centric
        self.is_add_bottom = cfg.is_add_bottom
        self.downsample_method = cfg.downsample_method
        self.pcd_noise = cfg.get('pcd_noise', 0)

        self.num_eef = cfg.num_eef
        self.dof = cfg.dof
        self.dataset_type = cfg.dataset_type

        self.eef_representation = cfg.eef_representation
        self.original_gripper_pcd = np.array(cfg.original_gripper_pcd)

        ## also predict per-frame binary in-hand status alongside gripper
        self.predict_in_hand = cfg.get('predict_in_hand', False)

        ## formerly hardcoded; overridable from the dataset yaml
        self.nbr_side_mapping = dict(cfg.get('nbr_side_mapping', DEFAULT_NBR_SIDE_MAPPING))
        self.visibility_search_window = cfg.get('visibility_search_window', DEFAULT_VISIBILITY_SEARCH_WINDOW)

        self.statistics = {}

        if mode == 'train':
            # Process the data
            print('Processing dataset...')
            self.process_select(cfg, **kwargs)
            self.skill_names = list(self.statistics['skill_embs_all_tasks'].keys())
            self.task_names = list(self.statistics['task_emb_dict'].keys())
        else:
            self.data = None
            self.normalizer = None

    @property
    def raw_file_names(self):
        return os.listdir(os.path.join(self.root, 'raw'))

    @property
    def processed_file_path(self):
        return os.path.join(self.root, 'processed', f'{self.dataset_type}.pt')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def process_select(self, cfg, **kwargs):
        if self.dataset_type == 'per_skill_dmg_traj':
            ## dexmimicgen
            self.data = self.process_per_skill_dmg_traj(cfg, **kwargs)
            self.normalizer = self.get_normalizer_and_statistics(self.data)
        elif self.dataset_type == 'per_skill_biop_jpose':
            self.data = self.process_per_biop(cfg, **kwargs)
            self.normalizer = self.get_normalizer_and_statistics(self.data, mode='jpose')
        else:
            raise NotImplementedError(f'Dataset type {self.dataset_type} not implemented!')

    ## ------------------------------------------------------------------
    ## shared hdf5 processing skeleton
    ## ------------------------------------------------------------------

    @staticmethod
    def _select_demos(f, cfg):
        """Sorted demo names, truncated to cfg.n_use if configured."""
        demos = [ent for ent in f['data'].keys() if ent.startswith('demo_')]
        inds = sorted(int(demo.split('_')[-1]) for demo in demos)
        demos = [f'demo_{ind}' for ind in inds]

        n_use = cfg.n_use if 'n_use' in cfg else len(demos)
        if n_use > len(demos):
            print(f'n_use {n_use} > len(demos) {len(demos)}')
            n_use = len(demos)
        return demos[:n_use], inds[:n_use]

    @staticmethod
    def _scan_interested(f, demos, skill_matcher, first_match_only=False):
        """Collect skill names matching skill_matcher and their first related obj."""
        interested_objs = set()
        interested_skills = set()
        for demo in demos:
            sg_info = f[f'data/{demo}/sg_info']
            for skill_name in sg_info.keys():
                if skill_matcher(skill_name):
                    interested_skills.add(skill_name)
                    interested_objs.add(sg_info[skill_name]['related_objs'][0].decode('utf-8'))
                    if first_match_only:
                        break
        return interested_objs, interested_skills

    def _process_hdf5_files(self, cfg, skill_matcher, demo_slices, first_match_only=False):
        """Shared iteration over raw hdf5 files for all per-skill dataset types.

        demo_slices(f, demo, robot_names, task_name, interested_objs,
                    interested_skills) -> list of data slices for one demo
        (including any aug_traj_nums repetition).
        """
        print('Processing hdf5 dataset...')
        data_list = []
        cache_dir = os.path.join(EQUIV_PRIMITIVE_PATH, cfg.embedding_cache_dir)

        self.involved_skill_names = set()
        skill_embs_all_tasks = {}
        matched_action_sgs = {}
        involved_tasks = set()

        for file_name in self.raw_file_names:
            if 'hdf5' not in file_name:
                continue

            hdf5_path = os.path.join(self.root, 'raw', file_name)
            with h5py.File(hdf5_path, 'r') as f:
                ## read sg
                sg_params = json.loads(f['sg_params'][()].decode('utf-8'))
                robot_names = sg_params['robots']
                task_name = sg_params['task_name']
                involved_tasks.add(task_name)

                demos, inds = self._select_demos(f, cfg)

                ## record the skillwise_sgs
                matched_action_sgs[task_name] = f[f'data/demo_{inds[0]}/matched_actions_json'][()]

                ## interested objs and skills for each task
                interested_objs, interested_skills = self._scan_interested(
                    f, demos, skill_matcher, first_match_only=first_match_only)

                skill_name_to_emb = get_embs_without_saving(list(interested_skills), cache_dir=cache_dir)
                skill_embs_all_tasks.update(skill_name_to_emb)
                self.involved_skill_names |= interested_skills

                for demo in demos:
                    data_list.extend(demo_slices(
                        f, demo, robot_names, task_name, interested_objs, interested_skills))

        os.makedirs(os.path.join(self.root, 'processed'), exist_ok=True)
        torch.save((data_list, None), self.processed_file_path)
        print('processed all hdf5 files!')

        cache_name = f'{cfg.dataset_type}_skill_name_to_emb.npy'
        save_embs(skill_embs_all_tasks, cache_dir=cache_dir, cache_name=cache_name)

        task_emb_dict = get_embs_without_saving(list(involved_tasks), cache_dir=cache_dir)
        self.statistics['task_emb_dict'] = task_emb_dict
        self.statistics['skill_embs_all_tasks'] = skill_embs_all_tasks
        self.statistics['matched_action_sgs'] = matched_action_sgs

        return data_list

    ## ------------------------------------------------------------------
    ## per_skill_biop_jpose
    ## ------------------------------------------------------------------

    def process_per_biop(self, cfg, **kwargs):
        primitive_kws = cfg.uniskills

        def demo_slices(f, demo, robot_names, task_name, interested_objs, interested_skills):
            sg_info = f[f'data/{demo}/sg_info']
            rbt_states = get_rbt_states(f[f'data/{demo}/obs'], robot_names)

            slices = []
            for _ in range(cfg.aug_traj_nums):
                for skill_name, skill_info in sg_info.items():
                    ## only use bimanual skills
                    if BIMANUAL_KW not in skill_name:
                        continue
                    slices.append(self.get_dataslice_bimanual_jpose(
                        skill_info, skill_name, rbt_states, task_name))
            return slices

        return self._process_hdf5_files(
            cfg,
            skill_matcher=lambda name: any(kw in name for kw in primitive_kws),
            demo_slices=demo_slices,
            first_match_only=True,
        )

    def get_dataslice_bimanual_jpose(self, skill_info, skill_name, rbt_states, task_name):
        pre_sg = get_sg(skill_info, 'pre_sg')
        qtraj_indice = pre_sg.graph['idx_list']
        selected_jpose = np.concatenate(
            [rbt_states['robot0_joint_pos'][qtraj_indice],
             rbt_states['robot1_joint_pos'][qtraj_indice]],
            axis=0,
        ).astype(np.float32)

        return {
            'jpose': selected_jpose,
            'skill_name': str_to_ascii_tensor(skill_name),
            'task_name': str_to_ascii_tensor(task_name),
        }

    ## ------------------------------------------------------------------
    ## per_skill_dmg_traj
    ## ------------------------------------------------------------------

    def process_per_skill_dmg_traj(self, cfg, **kwargs):
        traj_len = cfg.pred_horizon
        primitive_kws = cfg.uniskills

        def match_skill_key(skill_name):
            return next((kw for kw in primitive_kws if kw in skill_name), None)

        def demo_slices(f, demo, robot_names, task_name, interested_objs, interested_skills):
            sg_info = f[f'data/{demo}/sg_info']
            obs_grp = f[f'data/{demo}/obs']
            rbt_states = get_rbt_states(obs_grp, robot_names)
            obj_pcds = get_pc_instances(obs_grp, interested_objs)
            rbt_action = get_rbt_actions(f[f'data/{demo}/actions'][()], robot_names)
            obj_visibility = get_obj_visibility(obs_grp, interested_objs)

            slices = []
            for _ in range(cfg.aug_traj_nums):
                # Create separate data slices for each skill name
                for skill_name in interested_skills:
                    skill_info = sg_info[skill_name]
                    if BIMANUAL_KW in skill_name:
                        data_slice = self.get_dataslice_bimanual_kp(
                            skill_info, skill_name, obj_pcds, traj_len,
                            rbt_states, rbt_action, task_name)
                    else:
                        data_slice = self.get_dataslice_unimanual(
                            skill_info, skill_name, match_skill_key(skill_name),
                            cfg, traj_len, obj_pcds, obj_visibility,
                            rbt_states, rbt_action, task_name)
                    slices.append(data_slice)
            return slices

        return self._process_hdf5_files(
            cfg,
            skill_matcher=lambda name: match_skill_key(name) is not None,
            demo_slices=demo_slices,
        )

    def get_bimanual_pcs(self, pre_sg, related_pc_dict, observation_idx, num_points):
        if len(related_pc_dict) != len(self.nbr_side_mapping):
            raise ValueError(
                f'Expected {len(self.nbr_side_mapping)} pc instances '
                f'(one per {list(self.nbr_side_mapping)}), got {len(related_pc_dict)}')

        bimanual_pcs = {pc_kw: None for pc_kw in self.nbr_side_mapping.values()}
        table_pc_offset = None
        for obj_name, obj_pc_list in related_pc_dict.items():
            obj_pc = obj_pc_list[observation_idx]
            obj_nbr = list(pre_sg.neighbors(obj_name))[0]
            pc_kw = self.nbr_side_mapping[obj_nbr]

            obj_pc_tensor, obj_offset = self.get_obj_pc_tensor(obj_pc, num_points)
            bimanual_pcs[pc_kw] = obj_pc_tensor
            if pc_kw == 'pc':
                table_pc_offset = obj_offset

        return bimanual_pcs, table_pc_offset

    ## note that this dual_manual dataset cannot merge with unimanual dataset
    def get_dataslice_bimanual_kp(self, skill_info, skill_name, obj_pcds, traj_len, rbt_states, rbt_action, task_name):
        data_slice = {}

        pre_sg = get_sg(skill_info, 'pre_sg')

        ## NOTE: we cannot use pre_sg.graph['idx_list'] because it has been modified in unimanual skill detection.
        extended_ids = skill_info['extended_ids'][()]

        ## obtain the pc at the first frame of the skill window
        observation_idx = extended_ids[0]
        bimanual_pcs, init_pc_offset = self.get_bimanual_pcs(pre_sg, obj_pcds, observation_idx, self.pc_shape[0])
        for pc_kw, pc_tensor in bimanual_pcs.items():
            data_slice[pc_kw] = pc_tensor.to(torch.float32).reshape(1, -1, 3)

        ## random select one eefpose at the switch point
        random_switch_id = np.random.choice(extended_ids)
        eef_trans = {}
        for side in ('robot0', 'robot1'):
            trans = compose_transformation(
                rbt_states[f'{side}_eef_pos'][random_switch_id],
                rbt_states[f'{side}_eef_quat'][random_switch_id])
            eef_trans[side] = centralize_grasp(trans, init_pc_offset)

        ## make the bikp compatible to the unimanual dataset
        rep_times = traj_len // 2
        pre_dual_eef = np.concatenate(
            [np.tile(eef_trans['robot0'], (rep_times, 1, 1)),
             np.tile(eef_trans['robot1'], (rep_times, 1, 1))], axis=0)

        ## gripper action, won't be used
        gripper_list = np.concatenate(
            [np.tile(rbt_action['robot0'][random_switch_id], (rep_times, 1, 1)),
             np.tile(rbt_action['robot1'][random_switch_id], (rep_times, 1, 1))], axis=0)

        data_slice['eefpos'] = torch.tensor(pre_dual_eef).to(torch.float32)
        data_slice['gripper'] = torch.tensor(gripper_list).to(torch.float32)
        data_slice['skill_name'] = str_to_ascii_tensor(skill_name)
        data_slice['task_name'] = str_to_ascii_tensor(task_name)

        return data_slice

    def get_obj_pc_tensor(self, obj_pc_raw, num_points):
        pc_dim = obj_pc_raw.shape[1]

        obj_pc_n, obj_offset = centralize_downsample(
            obj_pc_raw,
            (num_points, pc_dim),
            obj_centric=self.is_obj_centric,
            add_bottom=self.is_add_bottom,
            method=self.downsample_method,
            debug_visualize=False
        )
        if self.pcd_noise > 0:
            obj_pc_n = add_pcd_noise(obj_pc_n, self.pcd_noise)

        # Reshape to (1, num_points, channels)
        obj_pc_tensor = torch.tensor(obj_pc_n).unsqueeze(0).to(torch.float32)
        return obj_pc_tensor, obj_offset

    def _first_visible_idx(self, obj_visibility, obj_name, start_idx, task_name):
        """First frame in the search window where obj_name is visible."""
        for obj_pc_idx in range(start_idx, start_idx + self.visibility_search_window):
            if obj_visibility[obj_name][obj_pc_idx] == 1:
                return obj_pc_idx
        raise ValueError(f'No valid object visibility found for {obj_name} in {task_name}')

    def get_dataslice_unimanual(self, skill_info, skill_name, skill_key, cfg, traj_len, obj_pcds, obj_visibility, rbt_states, rbt_action, task_name):
        data_slice = {}

        pre_sg = get_sg(skill_info, 'pre_sg')

        related_objs = skill_info['related_objs'][()]
        obj_name = related_objs[0].decode('utf-8')
        rbt_name = skill_info['related_rbts'][0].decode('utf-8')
        idx_list = skill_info['extended_ids'][()]
        essential_ids = skill_info['essential_ids'][()]

        pre_start_idx = pre_sg.graph['idx_list'][0]

        obj_pc_idx = self._first_visible_idx(obj_visibility, obj_name, pre_start_idx, task_name)
        obj_pc_tensor, obj_offset = self.get_obj_pc_tensor(obj_pcds[obj_name][obj_pc_idx], cfg.num_points)

        ## if more than one objs, the second one is the in-hand obj
        in_hand_obj_name = related_objs[1].decode('utf-8') if len(related_objs) > 1 else None
        if in_hand_obj_name is not None:
            in_hand_idx = self._first_visible_idx(obj_visibility, in_hand_obj_name, pre_start_idx, task_name)
            in_hand_obj_pc_tensor, _ = self.get_obj_pc_tensor(obj_pcds[in_hand_obj_name][in_hand_idx], cfg.num_points)
        else:
            # Create a randomized point cloud with the same per-dimension mean and variance as obj_pc_tensor
            in_hand_obj_pc_tensor = get_random_in_hand_pc(obj_pc_tensor)
        ## note that in_hand pc has been centered

        if cfg.choose_id_method == 'rdp':
            chosen_ids = choose_ids_rdp(rbt_states[f'{rbt_name}_eef_pos'], traj_len, idx_list, essential_ids=essential_ids)
        else:
            chosen_ids = choose_ids(traj_len, idx_list, essential_ids, skill_key)
        eef_pos_list = rbt_states[f'{rbt_name}_eef_pos'][chosen_ids]
        eef_quat_list = rbt_states[f'{rbt_name}_eef_quat'][chosen_ids]
        eef_pos_list = list(map(compose_transformation, eef_pos_list, eef_quat_list))
        normalized_eef_pos_list = list(map(centralize_grasp, eef_pos_list, [obj_offset] * traj_len))
        normalized_eef_pos_tensor = torch.tensor(normalized_eef_pos_list).to(torch.float32).reshape(traj_len, 4, 4)

        gripper_list = rbt_action[rbt_name][chosen_ids]

        ## input
        data_slice['pc'] = obj_pc_tensor
        data_slice['in_hand_pc'] = in_hand_obj_pc_tensor
        ## output
        data_slice['eefpos'] = normalized_eef_pos_tensor
        data_slice['gripper'] = torch.tensor(gripper_list).to(torch.float32).reshape(traj_len, 1, 1)

        if self.predict_in_hand:
            if 'grasp_event_boundary' not in skill_info or skill_info['grasp_event_boundary'][()] is None:
                raise ValueError(f'grasp_event_boundary missing/None for {skill_name} ({task_name})')
            geb = skill_info['grasp_event_boundary'][()]
            in_hand_list = (np.asarray(chosen_ids) >= geb).astype(np.float32)
            data_slice['in_hand'] = torch.tensor(in_hand_list).to(torch.float32).reshape(traj_len, 1, 1)

        data_slice['skill_name'] = str_to_ascii_tensor(skill_name)
        data_slice['task_name'] = str_to_ascii_tensor(task_name)
        ## note: if rotation, then the min xy and max xy will be same. So we need mean instead of min/max
        if cfg.rot_aug:
            data_slice = rotate_dataslice(data_slice)

        return data_slice

    ## ------------------------------------------------------------------
    ## normalization
    ## ------------------------------------------------------------------

    def _pc_xyz(self, pc_arr, name):
        """xyz channels of a pc array; stats are computed on xyz only even with color."""
        if self.use_pc_color and pc_arr.shape[-1] == 6:
            return pc_arr[..., :3]
        if pc_arr.shape[-1] == 3:
            return pc_arr
        raise ValueError(f'Invalid {name} shape: {pc_arr.shape}')

    def get_normalizer_and_statistics(self, data_list, mode='unimanual'):
        normalizer = LinearNormalizer()

        if mode == 'jpose':
            jpose_arr = np.concatenate([data['jpose'].reshape(2, -1) for data in data_list], axis=0)
            jpose_stats = to_torch_stats(jpose_arr.reshape(-1, jpose_arr.shape[-1]))
            normalizer['jpose'] = get_torch_range_symmetric_normalizer_from_stat(jpose_stats)
            return normalizer

        ### normalize pc
        pc_arr = np.concatenate([data['pc'] for data in data_list], axis=0)
        pcd_stats = to_torch_stats(self._pc_xyz(pc_arr, 'pc').reshape(-1, 3))
        normalizer['pc'] = get_torch_isotropic_xyz_normalizer_from_stat(pcd_stats)

        ## normalize eefpos. first convert to 3vec or 4pts
        eef_pos_arr = np.concatenate([data['eefpos'] for data in data_list], axis=0)
        eef_pos_torch = torch.tensor(eef_pos_arr).to(torch.float32)

        if self.eef_representation == '3vec':
            eef_xyz_raw, _, _ = convert_trans_to_vec(eef_pos_torch.reshape(-1, 1, 4, 4))
            eef_xyz_np = eef_xyz_raw.detach().cpu().numpy()
            eef_stats = to_torch_stats(eef_xyz_np.reshape(-1, eef_xyz_np.shape[-1]))
        elif self.eef_representation == '4pts':
            eef_4pts_raw = convert_trans_to_4pts(eef_pos_torch.reshape(-1, 1, 4, 4), self.original_gripper_pcd)
            eef_4pts_np = eef_4pts_raw.detach().cpu().numpy()
            eef_stats = to_torch_stats(eef_4pts_np.reshape(-1, eef_4pts_np.shape[-1]))
        else:
            raise ValueError(f'Unsupported eef_representation: {self.eef_representation}')
        ## isotropic scale, zero offset — required for rotation equivariance
        normalizer['eefpos'] = get_torch_isotropic_xyz_normalizer_from_stat(eef_stats)

        ## normalize gripper
        gripper_arr = np.concatenate([data['gripper'] for data in data_list], axis=0)
        gripper_stats = to_torch_stats(gripper_arr.reshape(-1, gripper_arr.shape[-1]))
        normalizer['gripper'] = get_torch_range_symmetric_normalizer_from_stat(gripper_stats)

        ## normalize in-hand status separately (binary {0,1} -> ~{-1,1}), if present
        if 'in_hand' in data_list[0]:
            in_hand_arr = np.concatenate([data['in_hand'] for data in data_list], axis=0)
            in_hand_stats = to_torch_stats(in_hand_arr.reshape(-1, in_hand_arr.shape[-1]))
            normalizer['in_hand'] = get_torch_range_symmetric_normalizer_from_stat(in_hand_stats)

        ## set_scale.
        # For scale computation, use only xyz (first 3 channels) even if color is present
        pc_arr_for_scale = pc_arr[..., :3] if pc_arr.shape[-1] == 6 else pc_arr
        self.statistics['pc_scale'] = self.get_pc_scale(pc_arr_for_scale, eef_stats['max'].max())

        if 'in_hand_pc' in data_list[0]:
            in_hand_pc_arr = np.concatenate([data['in_hand_pc'] for data in data_list], axis=0)
            in_hand_pcd_stats = to_torch_stats(self._pc_xyz(in_hand_pc_arr, 'in_hand_pc').reshape(-1, 3))
            normalizer['in_hand_pc'] = get_torch_isotropic_xyz_normalizer_from_stat(in_hand_pcd_stats)

        return normalizer

    def get_pc_scale(self, pc_data, ac_scale):
        """
        pc_data: (N_demos*demo_len, num_points, 3)
        """
        centroid = pc_data.mean(axis=1, keepdims=True)
        centered_pc = pc_data - centroid
        pc_scale = np.linalg.norm(centered_pc, axis=-1).mean()
        normed_pc_scale = pc_scale / ac_scale
        return normed_pc_scale
