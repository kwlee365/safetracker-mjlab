"""Tests for the HuB teacher task."""

from typing import Any
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.hub.mdp import (
  HubMotionCommand,
  HubMotionCommandCfg,
  push_curriculum,
)
from mjlab.tasks.registry import load_env_cfg

TASK_ID = "Mjlab-HuB-Teacher-Unitree-G1"


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def motion_file(tmp_path_factory) -> str:
  """Static G1 motion: 10 frames standing, then 10 with the right leg lifted."""
  model = Entity(get_g1_robot_cfg()).compile()
  data = mujoco.MjData(model)
  num_frames = 20
  joint_pos = np.zeros((num_frames, model.nq - 7))
  joint_pos[10:, model.joint("right_hip_pitch_joint").qposadr[0] - 7] = -0.8
  joint_pos[10:, model.joint("right_knee_joint").qposadr[0] - 7] = 1.2
  body_pos = np.zeros((num_frames, model.nbody - 1, 3))
  body_quat = np.zeros((num_frames, model.nbody - 1, 4))
  for t in range(num_frames):
    data.qpos[:7] = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = joint_pos[t]
    mujoco.mj_kinematics(model, data)
    body_pos[t] = data.xpos[1:]
    body_quat[t] = data.xquat[1:]
  path = tmp_path_factory.mktemp("hub") / "motion.npz"
  np.savez(
    path,
    fps=[50],
    joint_pos=joint_pos,
    joint_vel=np.zeros_like(joint_pos),
    body_pos_w=body_pos,
    body_quat_w=body_quat,
    body_lin_vel_w=np.zeros_like(body_pos),
    body_ang_vel_w=np.zeros_like(body_pos),
  )
  return str(path)


def test_hub_teacher_observations_contacts_and_torque_noise(motion_file, device):
  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = 2
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, HubMotionCommandCfg)
  motion_cmd.motion_file = motion_file
  env = ManagerBasedRlEnv(cfg, device=device)
  obs, _ = env.reset()

  # Table 3 of the paper: 1196 privileged teacher observations.
  for group in ("actor", "critic"):
    group_obs = obs[group]
    assert isinstance(group_obs, torch.Tensor)
    assert group_obs.shape == (2, 1196)

  command = env.command_manager.get_term("motion")
  assert isinstance(command, HubMotionCommand)
  command.time_steps[:] = torch.tensor([5, 15], device=env.device)
  assert command.ref_foot_contact.tolist() == [[True, True], [True, False]]
  assert command.ref_single_support.tolist() == [False, True]

  env.step(torch.zeros(2, env.action_manager.total_action_dim, device=env.device))
  # Random torques are bounded by 0.1 of the largest G1 effort limit (139 Nm).
  qfrc = env.sim.data.qfrc_applied
  assert 0.0 < qfrc.abs().max() <= 0.1 * 139.0
  env.close()


def test_push_curriculum_follows_episode_length():
  push_cfg = Mock(params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}})
  env = Mock(max_episode_length=500, common_step_counter=0)
  env.event_manager.get_term_cfg.return_value = push_cfg
  params: dict[str, Any] = {
    "event_name": "push_robot",
    "initial_scale": 0.2,
    "scale_step": 0.1,
    "promote_ratio": 0.5,
    "demote_ratio": 0.25,
    "update_interval": 10,
  }
  term = push_curriculum(Mock(params=params), env)
  env_ids = torch.arange(4)

  def end_episodes(step: int, length: int) -> tuple[float, float]:
    env.common_step_counter = step
    env.episode_length_buf = torch.full((4,), length)
    term(env, env_ids, **params)
    return push_cfg.params["velocity_range"]["x"]

  assert push_cfg.params["velocity_range"]["x"] == pytest.approx((-0.1, 0.1))
  assert end_episodes(10, 400) == pytest.approx((-0.15, 0.15))
  assert end_episodes(20, 50) == pytest.approx((-0.1, 0.1))
  assert end_episodes(30, 50) == pytest.approx((-0.1, 0.1))  # Floor.
  for step in range(40, 200, 10):
    end_episodes(step, 400)
  assert push_cfg.params["velocity_range"]["x"] == pytest.approx((-0.5, 0.5))


def test_hub_teacher_play_disables_perturbations():
  cfg = load_env_cfg(TASK_ID, play=True)
  assert "push_robot" not in cfg.events
  assert "torque_rfi" not in cfg.events
  assert "push" not in cfg.curriculum
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, HubMotionCommandCfg)
  assert motion_cmd.reference_offset_range == ((0.0, 0.0),) * 3
  assert motion_cmd.sampling_mode == "start"
