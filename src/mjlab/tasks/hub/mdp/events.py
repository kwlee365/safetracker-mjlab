from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class random_joint_torque:
  """Add random torques to the actuated joints (random force injection, RFI).

  Every environment step each actuated joint gets a torque drawn uniformly from
  ``[-scale, scale]`` times its actuator's effort limit, written to
  ``qfrc_applied`` and held for the physics substeps of the next step. This
  models unmodeled actuator dynamics (Campanaro et al., 2023).

  Use with ``mode="step"``. Requires joint-transmission actuators.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
    model = env.sim.mj_model
    ctrl_ids = asset.indexing.ctrl_ids.cpu().numpy()
    if np.any(model.actuator_trntype[ctrl_ids] != mujoco.mjtTrn.mjTRN_JOINT):
      raise ValueError("random_joint_torque requires joint-transmission actuators.")
    joint_ids = model.actuator_trnid[ctrl_ids, 0]
    self._dof_adr = torch.tensor(model.jnt_dofadr[joint_ids], device=env.device)
    self._effort_limit = torch.tensor(
      model.actuator_forcerange[ctrl_ids, 1], device=env.device, dtype=torch.float32
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    scale: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    del env_ids, asset_cfg  # Step events apply to every env.
    unit = 2.0 * torch.rand(env.num_envs, len(self._dof_adr), device=env.device) - 1.0
    env.sim.data.qfrc_applied[:, self._dof_adr] = unit * scale * self._effort_limit
