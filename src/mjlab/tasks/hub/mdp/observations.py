"""HuB teacher observations (Table 3 of the paper).

All quantities are expressed in the robot's heading frame: the frame at the
anchor (root) body with only its yaw. Rotations use the 6D representation of
the first two rotation matrix columns. Reference targets are the localized
ones of :class:`HubMotionCommand`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_inv,
  quat_mul,
  yaw_quat,
)

from .commands import HubMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> HubMotionCommand:
  return cast(HubMotionCommand, env.command_manager.get_term(command_name))


def _heading_inv(command: HubMotionCommand) -> torch.Tensor:
  """Inverse heading rotation broadcast over bodies, shape (B, N, 4)."""
  heading_inv = quat_inv(yaw_quat(command.robot_anchor_quat_w))
  return heading_inv[:, None, :].expand(-1, len(command.cfg.body_names), -1)


def _to_heading(command: HubMotionCommand, vec_w: torch.Tensor) -> torch.Tensor:
  return quat_apply(_heading_inv(command), vec_w).flatten(1)


def _rot6d(quat: torch.Tensor) -> torch.Tensor:
  return matrix_from_quat(quat)[..., :2].flatten(1)


def robot_body_pos_h(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Robot body positions relative to the root, without the root itself."""
  command = _command(env, command_name)
  pos = command.robot_body_pos_w - command.robot_anchor_pos_w[:, None, :]
  pos_h = quat_apply(_heading_inv(command), pos)
  keep = [
    i
    for i in range(len(command.cfg.body_names))
    if i != command.motion_anchor_body_index
  ]
  return pos_h[:, keep].flatten(1)


def robot_body_ori_h(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return _rot6d(quat_mul(_heading_inv(command), command.robot_body_quat_w))


def robot_body_lin_vel_h(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return _to_heading(command, command.robot_body_lin_vel_w)


def robot_body_ang_vel_h(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return _to_heading(command, command.robot_body_ang_vel_w)


def motion_body_pos_diff_h(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return _to_heading(command, command.body_pos_relative_w - command.robot_body_pos_w)


def motion_body_ori_diff_h(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Rotation from each robot body to its reference, in the heading frame."""
  command = _command(env, command_name)
  heading_inv = _heading_inv(command)
  diff_w = quat_mul(command.body_quat_relative_w, quat_inv(command.robot_body_quat_w))
  return _rot6d(quat_mul(quat_mul(heading_inv, diff_w), quat_inv(heading_inv)))


def motion_body_lin_vel_diff_h(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = _command(env, command_name)
  diff_w = command.body_lin_vel_relative_w - command.robot_body_lin_vel_w
  return _to_heading(command, diff_w)


def motion_body_ang_vel_diff_h(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = _command(env, command_name)
  diff_w = command.body_ang_vel_relative_w - command.robot_body_ang_vel_w
  return _to_heading(command, diff_w)


def motion_body_pos_h(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference body positions relative to the robot root."""
  command = _command(env, command_name)
  pos = command.body_pos_relative_w - command.robot_anchor_pos_w[:, None, :]
  return _to_heading(command, pos)


def motion_body_ori_h(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return _rot6d(quat_mul(_heading_inv(command), command.body_quat_relative_w))
