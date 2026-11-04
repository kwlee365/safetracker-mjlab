"""HuB teacher task configuration.

HuB: Learning Extreme Humanoid Balance (Zhang et al., CoRL 2025). The teacher is
trained with PPO on privileged observations (Table 3 of the paper) with the
rewards of Table 5 and the randomization of Table 6. Tracking is localized: the
reference is re-anchored to the robot's root position and heading every step.

Robot-specific settings (body, site and geom names) live in config/.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.hub import mdp
from mjlab.tasks.hub.mdp import HubMotionCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig


def _teacher_observation_terms() -> dict[str, ObservationTermCfg]:
  """Privileged teacher state (Table 3), all in the robot's heading frame."""
  motion = {"command_name": "motion"}
  return {
    "body_pos": ObservationTermCfg(func=mdp.robot_body_pos_h, params=motion),
    "body_ori": ObservationTermCfg(func=mdp.robot_body_ori_h, params=motion),
    "body_lin_vel": ObservationTermCfg(func=mdp.robot_body_lin_vel_h, params=motion),
    "body_ang_vel": ObservationTermCfg(func=mdp.robot_body_ang_vel_h, params=motion),
    "body_pos_diff": ObservationTermCfg(func=mdp.motion_body_pos_diff_h, params=motion),
    "body_ori_diff": ObservationTermCfg(func=mdp.motion_body_ori_diff_h, params=motion),
    "body_lin_vel_diff": ObservationTermCfg(
      func=mdp.motion_body_lin_vel_diff_h, params=motion
    ),
    "body_ang_vel_diff": ObservationTermCfg(
      func=mdp.motion_body_ang_vel_diff_h, params=motion
    ),
    "ref_body_pos": ObservationTermCfg(func=mdp.motion_body_pos_h, params=motion),
    "ref_body_ori": ObservationTermCfg(func=mdp.motion_body_ori_h, params=motion),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }


def make_hub_teacher_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base HuB teacher configuration."""

  ##
  # Observations
  ##

  # The teacher has no observation noise; HuB adds IMU noise to the student only.
  observations = {
    "actor": ObservationGroupCfg(
      terms=_teacher_observation_terms(),
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=_teacher_observation_terms(),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": HubMotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      joint_position_range=(-0.1, 0.1),
      reference_offset_range=((-0.02, 0.02), (-0.02, 0.02), (-0.1, 0.1)),
      # Override in robot cfg.
      motion_file="",
      anchor_body_name="",
      body_names=(),
      foot_body_names=("", ""),
    )
  }

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    # High-frequency push: a small velocity kick every second.
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 1.0),
      params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ),
    "torque_rfi": EventTermCfg(
      func=mdp.random_joint_torque,
      mode="step",
      params={"scale": 0.1},
    ),
    # HuB uses U(2.5, 3.5) in Isaac Gym, where PhysX averages it with the ground
    # friction of 1. Feet have contact priority here, so their friction applies
    # as is; this range matches the effective Isaac Gym friction.
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (1.75, 2.25),
        "shared_random": True,
      },
    ),
    "torso_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {0: (-0.1, 0.1), 1: (-0.1, 0.1), 2: (-0.1, 0.1)},
      },
    ),
    # Link mass x U(0.7, 1.3), scaling inertia along with it (mass ~ e^(2 alpha)).
    "link_mass": EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "alpha_range": (0.5 * math.log(0.7), 0.5 * math.log(1.3)),
      },
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.75, 1.25),
        "kd_range": (0.75, 1.25),
        "operation": "scale",
      },
    ),
  }

  ##
  # Rewards
  ##

  motion = {"command_name": "motion"}
  # Feet sites (left, right) and the tracked keypoints are set per robot.
  feet_sensor = {"sensor_name": "feet_ground_contact"}
  rewards: dict[str, RewardTermCfg] = {
    # Balance shaping.
    "com": RewardTermCfg(
      func=mdp.com_over_support_foot,
      weight=160.0,
      params={
        **motion,
        "std": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
    ),
    "foot_contact_mismatch": RewardTermCfg(
      func=mdp.foot_contact_mismatch,
      weight=-250.0,
      params={**motion, **feet_sensor},
    ),
    "close_feet": RewardTermCfg(
      func=mdp.feet_too_close,
      weight=-1000.0,
      params={
        "min_distance": 0.16,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
    ),
    # Tracking, with the relaxed body position tolerance.
    "body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=30.0,
      params={**motion, "std": 0.6, "body_names": None},
    ),
    "body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=20.0,
      params={**motion, "std": 0.3, "body_names": None},
    ),
    "body_lin_vel": RewardTermCfg(
      func=mdp.motion_relative_body_linear_velocity_error_exp,
      weight=8.0,
      params={**motion, "std": 3.0, "body_names": None},
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.motion_relative_body_angular_velocity_error_exp,
      weight=8.0,
      params={**motion, "std": 10.0, "body_names": None},
    ),
    "joint_pos": RewardTermCfg(
      func=mdp.motion_joint_position_error_exp,
      weight=32.0,
      params={**motion, "std": 0.7},
    ),
    "joint_vel": RewardTermCfg(
      func=mdp.motion_joint_velocity_error_exp,
      weight=16.0,
      params={**motion, "std": 10.0},
    ),
    # Limits and termination.
    "torque_limits": RewardTermCfg(
      func=mdp.joint_torque_limits, weight=-0.5, params={"soft_ratio": 0.9}
    ),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-30.0),
    "joint_vel_limits": RewardTermCfg(
      func=mdp.joint_velocity_limits,
      weight=-12.0,
      params={"limits": {}, "soft_ratio": 0.9},  # Limits set per robot.
    ),
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-60.0),
    # Regularization.
    "torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2.5e-5),
    "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-3),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-3e-6),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1.5),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time_at_landing,
      weight=250.0,
      params={**feet_sensor, "threshold": 0.25},
    ),
    "feet_contact_force": RewardTermCfg(
      func=mdp.feet_contact_force,
      weight=-0.2,
      params={**feet_sensor, "max_force": 500.0},
    ),
    "feet_stumble": RewardTermCfg(
      func=mdp.feet_stumble, weight=-3e-4, params={**feet_sensor, "ratio": 5.0}
    ),
    "feet_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-30.0,
      params={**feet_sensor, "asset_cfg": SceneEntityCfg("robot", site_names=())},
    ),
    "feet_orientation": RewardTermCfg(
      func=mdp.feet_not_flat,
      weight=-62.5,
      params={
        "height_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
    ),
    "feet_in_air": RewardTermCfg(
      func=mdp.feet_both_in_air, weight=-50.0, params=feet_sensor
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # Episodes end with the motion instead of looping it (see HubMotionCommand).
    "motion_end": TerminationTermCfg(func=mdp.motion_end, params=motion, time_out=True),
    "tracking_error": TerminationTermCfg(
      func=mdp.bad_mean_body_pos,
      params={**motion, "threshold": 0.5, "body_names": None},
    ),
  }

  ##
  # Curriculum
  ##

  # Pushes start at 0.1 m/s (HuB's evaluation push) and grow to the 0.5 m/s of
  # push_robot while the robot survives at least half of the motion it had left.
  curriculum: dict[str, CurriculumTermCfg] = {
    "push": CurriculumTermCfg(
      func=mdp.push_curriculum,
      params={
        "event_name": "push_robot",
        **motion,
        "initial_scale": 0.2,
        "scale_step": 0.1,
        "promote_ratio": 0.5,
        "demote_ratio": 0.25,
        "update_interval": 240,
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    # Longer than the HuB clips (up to 27 s refined), so an episode can play a
    # whole clip and ends at its last frame (motion_end).
    episode_length_s=30.0,
  )
