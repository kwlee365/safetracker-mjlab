"""HuB rewards (Table 5 of the paper).

Feet terms read a contact sensor whose primaries are the left and right foot,
in that order, with ``reduce="netforce"`` so forces are in the world frame. A
foot is in contact when its net contact force is at least ``force_threshold``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.tracking.mdp.rewards import _get_body_indexes
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values

from .commands import HubMotionCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _command(env: ManagerBasedRlEnv, command_name: str) -> HubMotionCommand:
  return cast(HubMotionCommand, env.command_manager.get_term(command_name))


def _feet_force(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  return sensor.data.force  # [B, 2, 3]


def _feet_in_contact(
  env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float
) -> torch.Tensor:
  return torch.norm(_feet_force(env, sensor_name), dim=-1) >= force_threshold


##
# Tracking.
##


def motion_relative_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _command(env, command_name)
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_relative_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _command(env, command_name)
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_relative_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.sum(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
  return torch.exp(-error / std**2)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.sum(torch.square(command.joint_vel - command.robot_joint_vel), dim=-1)
  return torch.exp(-error / std**2)


##
# Balance shaping.
##


def com_over_support_foot(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the ground-projected COM for staying over the support foot.

  Only active while the reference is in single support. ``asset_cfg.site_ids``
  selects the left and right sole sites; the support foot is the lower
  reference foot.
  """
  command = _command(env, command_name)
  asset: Entity = env.scene[asset_cfg.name]
  com_xy = asset.data.data.subtree_com[:, asset.data.indexing.root_body_id, :2]
  feet_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]
  support = command.ref_support_foot
  support_xy = feet_xy[torch.arange(env.num_envs, device=env.device), support]
  error = torch.sum(torch.square(com_xy - support_xy), dim=-1)
  return torch.exp(-error / std**2) * command.ref_single_support.float()


def foot_contact_mismatch(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Count feet whose contact state differs from the reference."""
  command = _command(env, command_name)
  contact = _feet_in_contact(env, sensor_name, force_threshold)
  return torch.sum(contact != command.ref_foot_contact, dim=-1).float()


def feet_too_close(
  env: ManagerBasedRlEnv,
  min_distance: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize the two feet (``asset_cfg.site_ids``) for being closer than
  ``min_distance``."""
  asset: Entity = env.scene[asset_cfg.name]
  feet = asset.data.site_pos_w[:, asset_cfg.site_ids]
  distance = torch.norm(feet[:, 0] - feet[:, 1], dim=-1)
  return torch.clamp(min_distance - distance, min=0.0)


##
# Limits.
##


class joint_torque_limits:
  """Penalize actuator forces beyond ``soft_ratio`` of the effort limit."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
    ctrl_ids = asset.indexing.ctrl_ids.cpu().numpy()
    self.effort_limit = torch.tensor(
      env.sim.mj_model.actuator_forcerange[ctrl_ids, 1],
      device=env.device,
      dtype=torch.float32,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    soft_ratio: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    excess = asset.data.actuator_force.abs() - soft_ratio * self.effort_limit
    return torch.sum(excess.clamp(min=0.0), dim=-1)


class joint_velocity_limits:
  """Penalize joint speeds beyond ``soft_ratio`` of the velocity limit.

  ``limits`` maps joint name patterns to velocity limits (rad/s). The excess of
  each joint is clipped to 1 rad/s.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
    _, _, limits = resolve_matching_names_values(
      data=cfg.params["limits"], list_of_strings=asset.joint_names
    )
    if len(limits) != len(asset.joint_names):
      raise ValueError("Every joint needs a velocity limit.")
    self.velocity_limit = torch.tensor(limits, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    limits: dict[str, float],
    soft_ratio: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del limits  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    excess = asset.data.joint_vel.abs() - soft_ratio * self.velocity_limit
    return torch.sum(excess.clamp(min=0.0, max=1.0), dim=-1)


##
# Feet.
##


def feet_air_time_at_landing(
  env: ManagerBasedRlEnv, sensor_name: str, threshold: float
) -> torch.Tensor:
  """Reward ``air time - threshold`` of each foot when it lands."""
  sensor: ContactSensor = env.scene[sensor_name]
  last_air_time = sensor.data.last_air_time
  assert last_air_time is not None
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  return torch.sum((last_air_time - threshold) * first_contact.float(), dim=-1)


def feet_contact_force(
  env: ManagerBasedRlEnv, sensor_name: str, max_force: float
) -> torch.Tensor:
  force = torch.norm(_feet_force(env, sensor_name), dim=-1)
  return torch.sum((force - max_force).clamp(min=0.0), dim=-1)


def feet_stumble(
  env: ManagerBasedRlEnv, sensor_name: str, ratio: float = 5.0
) -> torch.Tensor:
  """Penalize horizontal contact forces larger than ``ratio`` times the vertical."""
  force = _feet_force(env, sensor_name)
  horizontal = torch.norm(force[..., :2], dim=-1)
  return torch.any(horizontal > ratio * force[..., 2].abs(), dim=-1).float()


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize the squared velocity of feet (``asset_cfg.site_ids``) in contact."""
  asset: Entity = env.scene[asset_cfg.name]
  contact = _feet_in_contact(env, sensor_name, force_threshold)
  vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids]
  return torch.sum(torch.sum(torch.square(vel), dim=-1) * contact.float(), dim=-1)


def feet_not_flat(
  env: ManagerBasedRlEnv,
  height_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize tilted feet (``asset_cfg.site_ids``) close to the ground.

  The tilt is the horizontal part of gravity in the foot frame.
  """
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.site_quat_w[:, asset_cfg.site_ids]
  gravity = asset.data.gravity_vec_w[:, None, :].expand(-1, quat.shape[1], -1)
  tilt = torch.norm(quat_apply_inverse(quat, gravity)[..., :2], dim=-1)
  near_ground = asset.data.site_pos_w[:, asset_cfg.site_ids, 2] < height_threshold
  return torch.sum(tilt * near_ground.float(), dim=-1)


def feet_both_in_air(
  env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 1.0
) -> torch.Tensor:
  contact = _feet_in_contact(env, sensor_name, force_threshold)
  return (~contact.any(dim=-1)).float()
