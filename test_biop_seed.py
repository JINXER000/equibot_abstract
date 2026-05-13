import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch

from equibot.policies.aloha_wrapper import pddl_wrapper


def _install_lightweight_policy_import_stubs(monkeypatch):
    fake_normalizer = types.ModuleType("equibot.policies.utils.normalizer")

    class DummyLinearNormalizer:
        pass

    fake_normalizer.LinearNormalizer = DummyLinearNormalizer
    monkeypatch.setitem(sys.modules, fake_normalizer.__name__, fake_normalizer)

    fake_lan_utils = types.ModuleType("equibot.policies.utils.lan_utils")

    class DummyMLPEncoder:
        pass

    fake_lan_utils.get_and_save_skill_bert_embs = lambda *args, **kwargs: None
    fake_lan_utils.MLPEncoder = DummyMLPEncoder
    monkeypatch.setitem(sys.modules, fake_lan_utils.__name__, fake_lan_utils)

    fake_sim3_encoder = types.ModuleType("equibot.policies.vision.sim3_encoder")

    class DummySIM3Vec4Latent:
        pass

    fake_sim3_encoder.SIM3Vec4Latent = DummySIM3Vec4Latent
    monkeypatch.setitem(sys.modules, fake_sim3_encoder.__name__, fake_sim3_encoder)

    fake_conditional_unet = types.ModuleType(
        "equibot.policies.utils.equivariant_diffusion.conditional_unet1d"
    )

    class DummyVecConditionalUnet1D:
        pass

    class DummyFeatFusion:
        pass

    fake_conditional_unet.VecConditionalUnet1D = DummyVecConditionalUnet1D
    fake_conditional_unet.FeatFusion = DummyFeatFusion
    monkeypatch.setitem(
        sys.modules, fake_conditional_unet.__name__, fake_conditional_unet
    )

    fake_sdp_encoder = types.ModuleType("equibot.policies.vision.sdp_encoder")

    class DummySDPEncoder:
        pass

    fake_sdp_encoder.SDPEncoder = DummySDPEncoder
    monkeypatch.setitem(sys.modules, fake_sdp_encoder.__name__, fake_sdp_encoder)

    fake_ema_model = types.ModuleType("equibot.policies.utils.diffusion.ema_model")

    class DummyEMAModel:
        pass

    fake_ema_model.EMAModel = DummyEMAModel
    monkeypatch.setitem(sys.modules, fake_ema_model.__name__, fake_ema_model)

    fake_sdp_unet = types.ModuleType(
        "equibot.policies.utils.sdp_diffusion.irreps_conditional_unet1d"
    )

    class DummyIrrepConditionalUnet1D:
        pass

    fake_sdp_unet.IrrepConditionalUnet1D = DummyIrrepConditionalUnet1D
    monkeypatch.setitem(sys.modules, fake_sdp_unet.__name__, fake_sdp_unet)


def _load_biop_skill_policy(monkeypatch):
    module_name = "equibot.policies.agents.per_skill_policy"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    _install_lightweight_policy_import_stubs(monkeypatch)
    module = importlib.import_module(module_name)
    return module.BiopSkillPolicy


def _load_sdp_policy(monkeypatch):
    module_name = "equibot.policies.agents.sdp_policy"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    _install_lightweight_policy_import_stubs(monkeypatch)
    module = importlib.import_module(module_name)
    return module.SDPPolicy


class _FakeNoiseScheduler:
    def set_timesteps(self, num_diffusion_iters):
        self.timesteps = list(range(num_diffusion_iters - 1, -1, -1))

    def step(self, model_output, timestep, sample):
        return SimpleNamespace(prev_sample=sample)


class _FakeNoiseNet:
    def __call__(self, sample, timesteps):
        return torch.zeros_like(sample)


class _FakeSDPNoiseNet:
    def __call__(self, sample, timestep, global_cond):
        return torch.zeros_like(sample)


def _build_biop_policy(monkeypatch):
    biop_skill_policy_cls = _load_biop_skill_policy(monkeypatch)
    policy = biop_skill_policy_cls.__new__(biop_skill_policy_cls)
    policy.device = "cpu"
    policy.num_eef = 2
    policy.dof = 7
    policy.num_diffusion_iters = 4
    policy.noise_scheduler = _FakeNoiseScheduler()
    policy.ema = SimpleNamespace(
        averaged_model={"jpose_noise_pred_net": _FakeNoiseNet()}
    )
    policy.recover_jpose = lambda batch, key: batch.detach().cpu().numpy().reshape(
        -1, policy.num_eef, policy.dof
    )
    return biop_skill_policy_cls, policy


def _build_sdp_policy(monkeypatch):
    sdp_policy_cls = _load_sdp_policy(monkeypatch)
    policy = sdp_policy_cls.__new__(sdp_policy_cls)
    policy.device = "cpu"
    policy.obs_horizon = 1
    policy.pred_horizon = 4
    policy.action_dim = 10
    policy.num_diffusion_iters = 4
    policy.irrep_dim = 1
    policy.obs_as_global_cond = False
    policy.condition_type = "film"
    policy.eef_representation = "3vec"
    policy.noise_scheduler = _FakeNoiseScheduler()
    policy.ema = SimpleNamespace(
        averaged_model={"unitraj_noise_pred_net": _FakeSDPNoiseNet()}
    )
    policy.get_all_embs = lambda *args, **kwargs: torch.zeros(1, 1)
    policy.proc_pc = lambda *args, **kwargs: (torch.zeros(1, 1), None, None)
    policy.eef_recover_fn = lambda pred_eef_z, scale, center, key: (
        pred_eef_z.reshape(pred_eef_z.shape[0], pred_eef_z.shape[1], 3, 3),
        None,
        None,
    )
    policy.recover_gripper = lambda gripper, key: gripper
    return sdp_policy_cls, policy


def test_pred_bimanual_jposes_is_reproducible_for_same_seed(monkeypatch):
    biop_skill_policy_cls, policy = _build_biop_policy(monkeypatch)

    first_action, _ = biop_skill_policy_cls.pred_bimanual_jposes(
        policy, "bimanual", agent_obs=None, seed=123
    )
    second_action, _ = biop_skill_policy_cls.pred_bimanual_jposes(
        policy, "bimanual", agent_obs=None, seed=123
    )

    torch.testing.assert_close(first_action["jpose"], second_action["jpose"])


def test_pred_bimanual_jposes_changes_for_different_seeds(monkeypatch):
    biop_skill_policy_cls, policy = _build_biop_policy(monkeypatch)

    first_action, _ = biop_skill_policy_cls.pred_bimanual_jposes(
        policy, "bimanual", agent_obs=None, seed=123
    )
    second_action, _ = biop_skill_policy_cls.pred_bimanual_jposes(
        policy, "bimanual", agent_obs=None, seed=456
    )

    assert not torch.allclose(first_action["jpose"], second_action["jpose"])


def test_pred_bimanual_jposes_preserves_stochastic_default_behavior(monkeypatch):
    biop_skill_policy_cls, policy = _build_biop_policy(monkeypatch)
    torch.manual_seed(0)

    first_action, _ = biop_skill_policy_cls.pred_bimanual_jposes(
        policy, "bimanual", agent_obs=None
    )
    second_action, _ = biop_skill_policy_cls.pred_bimanual_jposes(
        policy, "bimanual", agent_obs=None
    )

    assert not torch.allclose(first_action["jpose"], second_action["jpose"])


def test_gen_uncond_jposes_forwards_seed_to_actor():
    recorded = {}

    def fake_pred_bimanual_jposes(skill_name, agent_obs=None, seed=None):
        recorded["skill_name"] = skill_name
        recorded["agent_obs"] = agent_obs
        recorded["seed"] = seed
        return {"jpose": torch.arange(14, dtype=torch.float32)}, {}

    wrapper = pddl_wrapper.__new__(pddl_wrapper)
    wrapper.agent = SimpleNamespace(
        actor=SimpleNamespace(pred_bimanual_jposes=fake_pred_bimanual_jposes)
    )

    jpose_out = wrapper.gen_uncond_jposes(None, None, "bimanual", seed=123)

    assert recorded == {"skill_name": "bimanual", "agent_obs": None, "seed": 123}
    torch.testing.assert_close(
        torch.from_numpy(jpose_out), torch.arange(14, dtype=torch.float32)
    )


def test_sdp_pred_unimanual_traj_is_reproducible_for_same_seed(monkeypatch):
    sdp_policy_cls, policy = _build_sdp_policy(monkeypatch)
    agent_obs = {"pc": torch.zeros(1, 1, 8, 3)}

    first_action, _ = sdp_policy_cls.pred_unimanual_traj(
        policy, "skill", agent_obs, task_name_batch="task", seed=123
    )
    second_action, _ = sdp_policy_cls.pred_unimanual_traj(
        policy, "skill", agent_obs, task_name_batch="task", seed=123
    )

    torch.testing.assert_close(first_action["eefpos"], second_action["eefpos"])
    torch.testing.assert_close(first_action["gripper"], second_action["gripper"])


def test_sdp_pred_unimanual_traj_changes_for_different_seeds(monkeypatch):
    sdp_policy_cls, policy = _build_sdp_policy(monkeypatch)
    agent_obs = {"pc": torch.zeros(1, 1, 8, 3)}

    first_action, _ = sdp_policy_cls.pred_unimanual_traj(
        policy, "skill", agent_obs, task_name_batch="task", seed=123
    )
    second_action, _ = sdp_policy_cls.pred_unimanual_traj(
        policy, "skill", agent_obs, task_name_batch="task", seed=456
    )

    assert not torch.allclose(first_action["eefpos"], second_action["eefpos"])


def test_sdp_pred_unimanual_traj_preserves_stochastic_default_behavior(monkeypatch):
    sdp_policy_cls, policy = _build_sdp_policy(monkeypatch)
    agent_obs = {"pc": torch.zeros(1, 1, 8, 3)}
    torch.manual_seed(0)

    first_action, _ = sdp_policy_cls.pred_unimanual_traj(
        policy, "skill", agent_obs, task_name_batch="task"
    )
    second_action, _ = sdp_policy_cls.pred_unimanual_traj(
        policy, "skill", agent_obs, task_name_batch="task"
    )

    assert not torch.allclose(first_action["eefpos"], second_action["eefpos"])


def test_gen_objcentric_traj_forwards_seed_to_actor(monkeypatch):
    recorded = {}

    def fake_pred_unimanual_traj(skill_name, agent_obs, task_name_batch=None, seed=None):
        recorded["skill_name"] = skill_name
        recorded["task_name_batch"] = task_name_batch
        recorded["seed"] = seed
        return {"eefpos": torch.eye(4), "gripper": torch.zeros(1)}, {}

    wrapper = pddl_wrapper.__new__(pddl_wrapper)
    wrapper.cfg = SimpleNamespace(
        device="cpu",
        data=SimpleNamespace(
            dataset=SimpleNamespace(is_obj_centric=True, downsample_method="fps")
        ),
    )
    wrapper.agent = SimpleNamespace(
        actor=SimpleNamespace(
            train=lambda training: None,
            pred_unimanual_traj=fake_pred_unimanual_traj,
        )
    )
    wrapper.centralize_obs = lambda *args, **kwargs: (
        {"pc": np.zeros((1, 1, 8, 3), dtype=np.float32)},
        {},
    )
    wrapper.decentralize_action = lambda action, offset: action

    action = wrapper.gen_objcentric_traj(
        "fallback_skill",
        {"pc": np.zeros((8, 3), dtype=np.float32)},
        skill_name="skill",
        task_name="task",
        seed=123,
    )

    assert recorded == {
        "skill_name": "skill",
        "task_name_batch": "task",
        "seed": 123,
    }
    torch.testing.assert_close(torch.from_numpy(action["grasp"]), torch.eye(4))


def test_gen_bimanual_kp_forwards_seed_to_actor(monkeypatch):
    recorded = {}

    def fake_pred_unimanual_traj(skill_name, agent_obs, task_name_batch=None, seed=None):
        recorded["skill_name"] = skill_name
        recorded["task_name_batch"] = task_name_batch
        recorded["seed"] = seed
        return {"eefpos": torch.eye(4), "gripper": torch.zeros(1)}, {}

    wrapper = pddl_wrapper.__new__(pddl_wrapper)
    wrapper.dataset = SimpleNamespace(
        pc_shape=(8, 3),
        is_obj_centric=True,
        is_add_bottom=False,
        downsample_method="fps",
    )
    wrapper.cfg = SimpleNamespace(device="cpu")
    wrapper.agent = SimpleNamespace(
        actor=SimpleNamespace(
            train=lambda training: None,
            pred_unimanual_traj=fake_pred_unimanual_traj,
        )
    )
    monkeypatch.setattr(
        "equibot.policies.aloha_wrapper.combined_pc_instances_and_offset",
        lambda *args, **kwargs: (
            np.zeros((1, 1, 8, 3), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
    )
    wrapper.decentralize_action = lambda action, offset: action

    action = wrapper.gen_bimanual_kp(
        {"pc": np.zeros((8, 3), dtype=np.float32)},
        skill_name="skill",
        task_name="task",
        seed=123,
    )

    assert recorded == {
        "skill_name": "skill",
        "task_name_batch": "task",
        "seed": 123,
    }
    torch.testing.assert_close(torch.from_numpy(action["eefpos"]), torch.eye(4))
