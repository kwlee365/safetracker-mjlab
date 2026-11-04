from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def bad_mean_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Terminate when the mean body position error exceeds ``threshold``."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return error.mean(dim=-1) > threshold


def motion_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """True on the last frame of the reference motion. Use with ``time_out=True``."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.time_steps >= command.motion.time_step_total - 1
