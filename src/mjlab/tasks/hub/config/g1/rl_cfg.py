"""RL configuration for the Unitree G1 HuB teacher."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def unitree_g1_hub_teacher_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO settings of HuB (Table 7).

  They match the G1 tracking settings except for the gradient clipping.
  """
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.algorithm.max_grad_norm = 0.2
  cfg.experiment_name = "g1_hub_teacher"
  cfg.wandb_project = "safetracker-hub"

  return cfg
