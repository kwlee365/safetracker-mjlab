from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class HubMotionCommand(MotionCommand):
  """Motion command with HuB's localized targets and reference foot contacts.

  On top of :class:`MotionCommand`, which already re-anchors the reference poses
  to the robot's anchor position and heading (``body_pos_relative_w``):

  - Reference velocities are rotated into the robot's heading the same way
    (``body_lin_vel_relative_w``, ``body_ang_vel_relative_w``).
  - A per-episode offset is added to the re-anchored reference positions, HuB's
    motion reference offset randomization. It does not move the reset state.
  - Reference foot contacts: both feet are grounded when their heights differ by
    less than ``contact_height_threshold``, otherwise only the lower foot is.
  """

  def __init__(self, cfg: HubMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.foot_indexes = [cfg.body_names.index(name) for name in cfg.foot_body_names]
    self.contact_height_threshold = cfg.contact_height_threshold
    self._reference_offset_range = torch.tensor(
      cfg.reference_offset_range, device=self.device
    )
    self.reference_offset = torch.zeros(self.num_envs, 3, device=self.device)
    self.body_lin_vel_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_ang_vel_relative_w = torch.zeros_like(self.body_lin_vel_relative_w)

  @property
  def ref_foot_height(self) -> torch.Tensor:
    """Reference heights of the left and right foot, shape (num_envs, 2)."""
    return self.motion.body_pos_w[self.time_steps][:, self.foot_indexes, 2]

  @property
  def ref_single_support(self) -> torch.Tensor:
    """Whether the reference stands on one foot, shape (num_envs,)."""
    height = self.ref_foot_height
    return (height[:, 0] - height[:, 1]).abs() >= self.contact_height_threshold

  @property
  def ref_support_foot(self) -> torch.Tensor:
    """Index of the lower reference foot (0 = left, 1 = right), shape (num_envs,)."""
    height = self.ref_foot_height
    return (height[:, 0] > height[:, 1]).long()

  @property
  def ref_foot_contact(self) -> torch.Tensor:
    """Reference contact state of the left and right foot, shape (num_envs, 2)."""
    lower = torch.nn.functional.one_hot(self.ref_support_foot, 2).bool()
    return torch.where(self.ref_single_support[:, None], lower, True)

  def update_relative_body_poses(self) -> None:
    super().update_relative_body_poses()
    self.body_pos_relative_w += self.reference_offset[:, None, :]
    delta_yaw = yaw_quat(
      quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w))
    )
    delta_yaw = delta_yaw[:, None, :].expand(-1, len(self.cfg.body_names), -1)
    self.body_lin_vel_relative_w = quat_apply(delta_yaw, self.body_lin_vel_w)
    self.body_ang_vel_relative_w = quat_apply(delta_yaw, self.body_ang_vel_w)

  def _resample_command(self, env_ids: torch.Tensor):
    super()._resample_command(env_ids)
    ranges = self._reference_offset_range
    self.reference_offset[env_ids] = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=self.device
    )


@dataclass(kw_only=True)
class HubMotionCommandCfg(MotionCommandCfg):
  foot_body_names: tuple[str, str]
  """Left and right foot bodies; their reference heights set the contacts."""
  contact_height_threshold: float = 0.05
  """Foot height difference above which the reference is in single support."""
  reference_offset_range: tuple[
    tuple[float, float], tuple[float, float], tuple[float, float]
  ] = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
  """Range of the per-episode xyz offset added to the reference positions."""

  def build(self, env: ManagerBasedRlEnv) -> HubMotionCommand:
    return HubMotionCommand(self, env)
