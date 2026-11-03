"""Unitree G1 HuB environment configurations."""

from dataclasses import replace

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.hub.hub_env_cfg import make_hub_teacher_env_cfg
from mjlab.tasks.hub.mdp import HubMotionCommandCfg

# All G1 bodies, in model order. The teacher observes every body.
G1_BODY_NAMES = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)

# The 12 tracked keypoints: hips, knees, ankles, shoulders, elbows and wrists.
G1_KEYPOINT_NAMES = (
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

# Motor velocity limits (rad/s). The waist and ankle linkages use 5020 motors.
G1_VELOCITY_LIMITS = {
  name: motor.velocity_limit
  for actuator, motor in (
    (g1.G1_ACTUATOR_5020, g1.ACTUATOR_5020),
    (g1.G1_ACTUATOR_7520_14, g1.ACTUATOR_7520_14),
    (g1.G1_ACTUATOR_7520_22, g1.ACTUATOR_7520_22),
    (g1.G1_ACTUATOR_4010, g1.ACTUATOR_4010),
    (g1.G1_ACTUATOR_WAIST, g1.ACTUATOR_5020),
    (g1.G1_ACTUATOR_ANKLE, g1.ACTUATOR_5020),
  )
  for name in actuator.target_names_expr
}

# Control delay of 20-60 ms in 5 ms physics steps, fixed for each episode.
CONTROL_DELAY_STEPS = (4, 12)


def _g1_robot_cfg_with_delay() -> EntityCfg:
  cfg = get_g1_robot_cfg()
  assert cfg.articulation is not None
  min_lag, max_lag = CONTROL_DELAY_STEPS
  actuators = tuple(
    replace(a, delay_min_lag=min_lag, delay_max_lag=max_lag, delay_hold_prob=1.0)
    for a in cfg.articulation.actuators
  )
  cfg.articulation = replace(cfg.articulation, actuators=actuators)
  return cfg


def unitree_g1_hub_teacher_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 HuB teacher configuration."""
  cfg = make_hub_teacher_env_cfg()

  cfg.scene.entities = {"robot": _g1_robot_cfg_with_delay()}
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (feet_ground_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, HubMotionCommandCfg)
  motion_cmd.anchor_body_name = "pelvis"
  motion_cmd.body_names = G1_BODY_NAMES
  motion_cmd.foot_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")

  for name in ("body_pos", "body_ori", "body_lin_vel", "body_ang_vel"):
    cfg.rewards[name].params["body_names"] = G1_KEYPOINT_NAMES
  for name in ("com", "close_feet", "feet_slip", "feet_orientation"):
    cfg.rewards[name].params["asset_cfg"].site_names = ("left_foot", "right_foot")
  cfg.rewards["joint_vel_limits"].params["limits"] = G1_VELOCITY_LIMITS
  cfg.terminations["tracking_error"].params["body_names"] = G1_KEYPOINT_NAMES

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["torso_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.viewer.body_name = "pelvis"

  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("torque_rfi", None)
    cfg.curriculum.pop("push", None)

    # Disable RSI and reference randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.reference_offset_range = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    motion_cmd.sampling_mode = "start"

  return cfg
