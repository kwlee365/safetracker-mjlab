from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class push_curriculum:
  """Scale a push event's velocity range with how long the robot survives.

  Each episode that ends scores the fraction it survived of the steps it had:
  until the end of the reference motion, at most the maximum episode length.
  Episodes start at any frame of the motion and end at its last one, so the
  episode length alone would not tell a fall from a late start.

  Pushes start at ``initial_scale`` times the range configured on the event.
  Every ``update_interval`` env steps the scale goes up by ``scale_step`` if the
  episodes that ended since the last update survived at least ``promote_ratio``
  on average, and down by ``scale_step`` if they survived less than
  ``demote_ratio``. The scale stays within ``[initial_scale, 1]``, so the
  event's own range is the final push.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._event_cfg = env.event_manager.get_term_cfg(cfg.params["event_name"])
    self._full_range: dict[str, tuple[float, float]] = dict(
      self._event_cfg.params["velocity_range"]
    )
    self._min_scale: float = cfg.params["initial_scale"]
    self.scale = self._min_scale
    self._survival_sum = 0.0
    self._num_episodes = 0
    self._last_update_step = 0
    self._apply_scale()

  def _apply_scale(self) -> None:
    self._event_cfg.params["velocity_range"] = {
      axis: (low * self.scale, high * self.scale)
      for axis, (low, high) in self._full_range.items()
    }

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    command_name: str,
    initial_scale: float,
    scale_step: float,
    promote_ratio: float,
    demote_ratio: float,
    update_interval: int,
  ) -> dict[str, torch.Tensor]:
    del event_name, initial_scale  # Resolved in __init__.
    # The first reset happens before any step, when no episode has ended.
    if env.common_step_counter > 0:
      # Curriculum terms run before the command resamples the reset envs, so
      # time_steps is still the frame each episode ended on.
      command = cast(MotionCommand, env.command_manager.get_term(command_name))
      length = env.episode_length_buf[env_ids].float()
      frames_left = command.motion.time_step_total - 1 - command.time_steps[env_ids]
      available = (length + frames_left).clamp(min=1, max=env.max_episode_length)
      self._survival_sum += float((length / available).clamp(max=1.0).sum())
      self._num_episodes += len(env_ids)

    steps_since_update = env.common_step_counter - self._last_update_step
    if steps_since_update >= update_interval and self._num_episodes > 0:
      ratio = self._survival_sum / self._num_episodes
      if ratio >= promote_ratio:
        self.scale = min(self.scale + scale_step, 1.0)
      elif ratio < demote_ratio:
        self.scale = max(self.scale - scale_step, self._min_scale)
      self._apply_scale()
      self._survival_sum = 0.0
      self._num_episodes = 0
      self._last_update_step = env.common_step_counter

    max_push = max(max(abs(low), abs(high)) for low, high in self._full_range.values())
    return {
      "scale": torch.tensor(self.scale),
      "max_push": torch.tensor(self.scale * max_push),
    }
