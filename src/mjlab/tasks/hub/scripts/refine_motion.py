"""Refine retargeted Unitree G1 reference motions for HuB balance training.

Implements the post-retargeting steps of HuB (Sec. 3.2 of the paper), after a
foot leveling step that is not part of HuB:

0. Foot leveling: feet on or near the ground get the ankle pitch and roll that
   make their soles horizontal. Retargeted clips often leave the support foot
   tilted by 10-20 degrees, so the robot would stand and land on an edge.
1. Grounded foot correction: during single support the root is translated so
   the support foot does not slide. Joint angles are left unchanged.
2. Center-of-mass filtering: clips whose ground-projected COM is further than
   ``com_threshold`` from the support foot during single support are rejected.
3. Transition stabilization: the first and last frames are repeated so the
   padding adds up to the duration of the balance phase.

Each frame is also put on the ground, so its lower sole touches it, which
assumes quasi-static clips without flight phases. Each clip is moved to start
at the origin. Inputs
are HuB/PHC motion pickles (joblib format) or mjlab motion CSVs. Each clip is
written as a CSV in the format of ``mjlab.scripts.csv_to_npz`` and, by
default, converted to npz with it.

Example:
  uv run -m mjlab.tasks.hub.scripts.refine_motion \\
    --input-path motions/hub --output-dir motions/hub_refined
"""

from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import tyro

import mjlab
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.scripts import csv_to_npz

# Joint order of HuB/PHC pickles and of mjlab motion CSVs.
G1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

# Index 0 is the left foot and index 1 the right foot everywhere below.
_SIDES = ("left", "right")


@dataclass(frozen=True)
class Motion:
  root_pos: np.ndarray
  """Root position in the world frame, shape (T, 3)."""
  root_quat: np.ndarray
  """Root orientation as a wxyz quaternion, shape (T, 4)."""
  joint_pos: np.ndarray
  """Joint positions in ``G1_JOINT_NAMES`` order, shape (T, 29)."""
  fps: float

  @property
  def num_frames(self) -> int:
    return self.root_pos.shape[0]


##
# Loading and saving.
##


class _NumpyArrayWrapper:
  """Stand-in for ``joblib.numpy_pickle.NumpyArrayWrapper``."""


def _reconstruct(subtype, shape, dtype) -> np.ndarray:
  return np.ndarray.__new__(subtype, shape, dtype)


def _scalar(dtype: np.dtype, data: bytes):
  return np.frombuffer(data, dtype=dtype, count=1)[0]


class _SafeUnpickler(pickle._Unpickler):
  """Unpickler that only reconstructs numpy arrays and scalars.

  Any other global is rejected, so loading a pickle cannot run arbitrary code.
  It also reads joblib pickles without joblib: joblib writes the raw bytes of
  each array right after the BUILD opcode of its wrapper.
  """

  _ALLOWED = {
    ("joblib.numpy_pickle", "NumpyArrayWrapper"): _NumpyArrayWrapper,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
    ("numpy.core.multiarray", "scalar"): _scalar,
    ("numpy._core.multiarray", "scalar"): _scalar,
  }

  def __init__(self, file) -> None:
    super().__init__(file)
    self._file = file

  def find_class(self, module: str, name: str):
    if (module, name) not in self._ALLOWED:
      raise pickle.UnpicklingError(f"Refusing to load global {module}.{name}")
    return self._ALLOWED[(module, name)]

  # The pure-Python unpickler internals used below are missing from typeshed.
  def _load_build(self) -> None:
    super().load_build()  # type: ignore[unresolved-attribute]
    stack: list = self.stack  # type: ignore[unresolved-attribute]
    if isinstance(stack[-1], _NumpyArrayWrapper):
      stack[-1] = self._read_joblib_array(vars(stack[-1]))

  dispatch = dict(pickle._Unpickler.dispatch)
  dispatch[pickle.BUILD[0]] = _load_build  # pyright: ignore[reportArgumentType]

  def _read_joblib_array(self, state: dict) -> np.ndarray:
    dtype: np.dtype = state["dtype"]
    if dtype.hasobject:
      raise pickle.UnpicklingError("Object arrays are not supported")
    if state.get("numpy_array_alignment_bytes") is not None:
      padding = int.from_bytes(self._file.read(1), "little")
      self._file.read(padding)
    shape = tuple(state["shape"])
    count = math.prod(shape)
    data = self._file.read(count * dtype.itemsize)
    array = np.frombuffer(data, dtype=dtype, count=count).copy()
    if state["order"] == "F":
      return array.reshape(shape[::-1]).T
    return array.reshape(shape)


def load_motion_pkl(path: str | Path) -> dict[str, Motion]:
  """Load the clips of a HuB/PHC motion pickle, keyed by clip name.

  Each clip holds ``root_trans_offset`` (T, 3), ``root_rot`` as xyzw (T, 4),
  ``dof`` (T, 29) and ``fps``.
  """
  with open(path, "rb") as f:
    clips = _SafeUnpickler(f).load()
  return {
    name: Motion(
      root_pos=np.asarray(clip["root_trans_offset"], dtype=np.float64),
      root_quat=np.asarray(clip["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]],
      joint_pos=np.asarray(clip["dof"], dtype=np.float64),
      fps=float(clip["fps"]),
    )
    for name, clip in clips.items()
  }


def load_motion_csv(path: str | Path, fps: float) -> Motion:
  """Load an mjlab motion CSV: root pos, root quat (xyzw), joint positions."""
  rows = np.loadtxt(path, delimiter=",", ndmin=2)
  return Motion(
    root_pos=rows[:, :3],
    root_quat=rows[:, [6, 3, 4, 5]],
    joint_pos=rows[:, 7:],
    fps=fps,
  )


def save_motion_csv(motion: Motion, path: str | Path) -> None:
  rows = np.concatenate(
    [motion.root_pos, motion.root_quat[:, [1, 2, 3, 0]], motion.joint_pos], axis=1
  )
  np.savetxt(path, rows, delimiter=",", fmt="%.8f")


##
# Kinematics.
##


@dataclass(frozen=True)
class FootKinematics:
  ankle_pos: np.ndarray
  """Ankle roll link positions, shape (T, 2, 3)."""
  foot_pos: np.ndarray
  """Sole center positions (``*_foot`` sites), shape (T, 2, 3)."""
  sole_height: np.ndarray
  """Height of the lowest point of each foot, shape (T, 2)."""
  foot_rot: np.ndarray
  """Ankle roll link orientations, shape (T, 2, 3, 3)."""
  com_pos: np.ndarray
  """Whole-body center of mass, shape (T, 3)."""


class G1Kinematics:
  """Forward kinematics of the mjlab Unitree G1 with CPU MuJoCo."""

  def __init__(self) -> None:
    self.model = Entity(get_g1_robot_cfg()).compile()
    self.data = mujoco.MjData(self.model)
    free_joint = int(
      np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    )
    self._free_adr = int(self.model.jnt_qposadr[free_joint])
    self._root_body = int(self.model.jnt_bodyid[free_joint])
    self._joint_adr = np.array(
      [self.model.joint(name).qposadr[0] for name in G1_JOINT_NAMES]
    )
    self._ankle_ids = [self.model.body(f"{s}_ankle_roll_link").id for s in _SIDES]
    self._site_ids = [self.model.site(f"{s}_foot").id for s in _SIDES]
    geom_names = [self.model.geom(i).name for i in range(self.model.ngeom)]
    self._foot_geoms = [
      [
        i
        for i, n in enumerate(geom_names)
        if re.fullmatch(rf"{s}_foot\d+_collision", n)
      ]
      for s in _SIDES
    ]
    for side, geoms in zip(_SIDES, self._foot_geoms, strict=True):
      types = self.model.geom_type[geoms]
      if not geoms or np.any(types != mujoco.mjtGeom.mjGEOM_CAPSULE):
        raise ValueError(f"Expected capsule collision geoms on the {side} foot")

    # Foot leveling solves for the ankle angles in closed form. It assumes a
    # pitch joint about y followed by a roll joint about x, on bodies that are
    # not rotated relative to their parents.
    # Indices in G1_JOINT_NAMES and limits of the ankle joints, indexed by
    # [foot, (pitch, roll)].
    self.ankle_joint_index = np.zeros((2, 2), dtype=np.int64)
    self.ankle_range = np.zeros((2, 2, 2))
    for k, side in enumerate(_SIDES):
      for j, (name, axis) in enumerate((("pitch", (0, 1, 0)), ("roll", (1, 0, 0)))):
        joint = self.model.joint(f"{side}_ankle_{name}_joint")
        body_quat = self.model.body_quat[self.model.jnt_bodyid[joint.id]]
        unrotated = np.allclose(body_quat, [1.0, 0.0, 0.0, 0.0])
        if not np.allclose(joint.axis, axis) or not unrotated:
          raise ValueError(f"Unexpected {side} ankle {name} joint for foot leveling")
        self.ankle_joint_index[k, j] = G1_JOINT_NAMES.index(joint.name)
        self.ankle_range[k, j] = joint.range
    # Depth of each sole below its ankle roll link when the foot is level.
    self.sole_depth = np.array([self._level_sole_depth(g) for g in self._foot_geoms])

  def _level_sole_depth(self, geoms: list[int]) -> float:
    lowest = np.inf
    mat = np.zeros(9)
    for g in geoms:
      mujoco.mju_quat2Mat(mat, self.model.geom_quat[g])
      radius, half_length = self.model.geom_size[g, :2]
      # mat[8] is the z component of the capsule axis in the body frame.
      low = self.model.geom_pos[g, 2] - abs(mat[8]) * half_length - radius
      lowest = min(lowest, low)
    return float(-lowest)

  def __call__(self, motion: Motion) -> FootKinematics:
    num_frames = motion.num_frames
    ankle_pos = np.zeros((num_frames, 2, 3))
    foot_pos = np.zeros((num_frames, 2, 3))
    sole_height = np.zeros((num_frames, 2))
    foot_rot = np.zeros((num_frames, 2, 3, 3))
    com_pos = np.zeros((num_frames, 3))
    qpos = self.data.qpos
    for t in range(num_frames):
      qpos[self._free_adr : self._free_adr + 3] = motion.root_pos[t]
      qpos[self._free_adr + 3 : self._free_adr + 7] = motion.root_quat[t]
      qpos[self._joint_adr] = motion.joint_pos[t]
      mujoco.mj_kinematics(self.model, self.data)
      mujoco.mj_comPos(self.model, self.data)
      ankle_pos[t] = self.data.xpos[self._ankle_ids]
      foot_pos[t] = self.data.site_xpos[self._site_ids]
      foot_rot[t] = self.data.xmat[self._ankle_ids].reshape(2, 3, 3)
      com_pos[t] = self.data.subtree_com[self._root_body]
      for k, geoms in enumerate(self._foot_geoms):
        sole_height[t, k] = self._lowest_capsule_point(geoms)
    return FootKinematics(ankle_pos, foot_pos, sole_height, foot_rot, com_pos)

  def _lowest_capsule_point(self, geoms: list[int]) -> float:
    center_z = self.data.geom_xpos[geoms, 2]
    # z component of each capsule axis (its local z) in the world frame.
    axis_z = self.data.geom_xmat[geoms, 8]
    radius = self.model.geom_size[geoms, 0]
    half_length = self.model.geom_size[geoms, 1]
    return float(np.min(center_z - np.abs(axis_z) * half_length - radius))


##
# Refinement steps.
##


def support_labels(
  ankle_z: np.ndarray, height_threshold: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
  """Label single-support frames and their support foot.

  Following HuB, both feet are grounded when their ankle heights differ by less
  than ``height_threshold``; otherwise only the lower foot is.

  Args:
    ankle_z: Ankle heights of the left and right foot, shape (T, 2).
    height_threshold: Height difference that separates single support.

  Returns:
    A boolean single-support mask and the support foot index (0 = left,
    1 = right) of each frame, both of shape (T,).
  """
  dz = ankle_z[:, 0] - ankle_z[:, 1]
  return np.abs(dz) >= height_threshold, (dz > 0).astype(np.int64)


def grounded_feet(ankle_z: np.ndarray, height_threshold: float = 0.05) -> np.ndarray:
  """Feet labeled as grounded by :func:`support_labels`, shape (T, 2)."""
  single, support = support_labels(ankle_z, height_threshold)
  return np.where(single[:, None], np.eye(2, dtype=bool)[support], True)


def foot_tilt(kin: FootKinematics) -> np.ndarray:
  """Angle between each sole normal and the vertical in degrees, shape (T, 2)."""
  return np.degrees(np.arccos(np.clip(kin.foot_rot[..., 2, 2], -1.0, 1.0)))


def _rot_x(angle: np.ndarray) -> np.ndarray:
  c, s, one, zero = np.cos(angle), np.sin(angle), np.ones_like(angle), 0 * angle
  rows = [[one, zero, zero], [zero, c, -s], [zero, s, c]]
  return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


def _rot_y(angle: np.ndarray) -> np.ndarray:
  c, s, one, zero = np.cos(angle), np.sin(angle), np.ones_like(angle), 0 * angle
  rows = [[c, zero, s], [zero, one, zero], [-s, zero, c]]
  return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


def level_ankle_angles(
  foot_rot: np.ndarray, pitch: np.ndarray, roll: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Ankle pitch and roll that make the soles horizontal.

  A foot's orientation is ``R_shin @ Ry(pitch) @ Rx(roll)``. The sole is level
  when ``Ry(p) @ Rx(r) @ z`` equals ``g = R_shin^T @ z``, the vertical in the
  shin frame, which gives ``r = -asin(g_y)`` and ``p = atan2(g_x, g_z)``. The
  foot keeps the heading set by the shin.

  Args:
    foot_rot: Foot orientations, shape (..., 3, 3).
    pitch: Current ankle pitch angles, shape (...).
    roll: Current ankle roll angles, shape (...).
  """
  ankle_rot = _rot_y(pitch) @ _rot_x(roll)
  shin_rot = foot_rot @ np.swapaxes(ankle_rot, -1, -2)
  g = shin_rot[..., 2, :]
  return np.arctan2(g[..., 0], g[..., 2]), -np.arcsin(np.clip(g[..., 1], -1.0, 1.0))


def level_feet(
  motion: Motion,
  kinematics: G1Kinematics,
  kin: FootKinematics,
  height_range: tuple[float, float] = (0.03, 0.10),
) -> Motion:
  """Level the soles of the feet that are on or near the ground.

  Each foot gets the ankle pitch and roll of :func:`level_ankle_angles`, so the
  rest of the leg is unchanged. The correction is blended by how high the level
  sole would be above the ground: full up to ``height_range[0]`` and none from
  ``height_range[1]``. Swing feet keep their reference orientation and the
  joint trajectories stay continuous. Angles are clipped to the joint limits.

  Args:
    motion: The clip to level.
    kinematics: Forward kinematics of the robot.
    kin: Forward kinematics of ``motion``.
    height_range: Sole heights (m) where the correction starts to fade and
      where it is gone.
  """
  # The ground is where the lower level sole usually is, as in grounding.
  level_sole_z = kin.ankle_pos[..., 2] - kinematics.sole_depth
  clearance = level_sole_z - np.median(level_sole_z.min(axis=1))
  low, high = height_range
  weight = np.clip((high - clearance) / (high - low), 0.0, 1.0)

  joint_pos = motion.joint_pos.copy()
  index = kinematics.ankle_joint_index
  pitch, roll = joint_pos[:, index[:, 0]], joint_pos[:, index[:, 1]]
  level = level_ankle_angles(kin.foot_rot, pitch, roll)
  for j, current in enumerate((pitch, roll)):
    blended = current + weight * (level[j] - current)
    for k in range(2):
      lower, upper = kinematics.ankle_range[k, j]
      joint_pos[:, index[k, j]] = np.clip(blended[:, k], lower, upper)
  return replace(motion, joint_pos=joint_pos)


def grounded_foot_offsets(
  ankle_pos: np.ndarray, single: np.ndarray, support: np.ndarray
) -> np.ndarray:
  """Root translations that keep the support foot static in single support.

  The horizontal position of the support ankle is pinned to where it is at the
  start of each single-support segment. The last offset carries over to the
  frames that follow, so the corrected root trajectory stays continuous.
  Height is left to grounding: pinning it would carry small height errors
  into every later frame.

  Returns:
    Per-frame root offsets with zero height, shape (T, 3).
  """
  offsets = np.zeros((ankle_pos.shape[0], 3))
  offset = np.zeros(2)
  anchor = np.zeros(2)
  for t in range(ankle_pos.shape[0]):
    if single[t]:
      foot = ankle_pos[t, support[t], :2]
      if t == 0 or not single[t - 1] or support[t - 1] != support[t]:
        anchor = foot + offset
      offset = anchor - foot
    offsets[t, :2] = offset
  return offsets


def static_end_frames(motion: Motion, atol: float = 1e-6) -> tuple[int, int]:
  """Count the copies of the first and last frame at either end of the clip."""
  state = np.concatenate([motion.root_pos, motion.root_quat, motion.joint_pos], axis=1)
  moving = np.abs(np.diff(state, axis=0)).max(axis=1) > atol
  if not moving.any():
    return motion.num_frames - 1, 0
  return int(np.argmax(moving)), int(np.argmax(moving[::-1]))


def transition_padding(motion: Motion, single: np.ndarray) -> tuple[int, int]:
  """Frames to prepend and append for transition stabilization.

  HuB repeats the first and last frames so that, together, they last as long
  as the balance phase. The balance phase is the span of single-support
  frames, or the moving part of the clip if there is none (e.g. squats).
  Copies already at either end count towards the target.
  """
  lead, trail = static_end_frames(motion)
  idx = np.flatnonzero(single)
  if idx.size > 0:
    balance_frames = int(idx[-1] - idx[0] + 1)
  else:
    balance_frames = motion.num_frames - lead - trail
  per_side = math.ceil(balance_frames / 2)
  return max(per_side - lead, 0), max(per_side - trail, 0)


def pad_motion(motion: Motion, lead: int, trail: int) -> Motion:
  def pad(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[:1].repeat(lead, 0), x, x[-1:].repeat(trail, 0)])

  return Motion(
    root_pos=pad(motion.root_pos),
    root_quat=pad(motion.root_quat),
    joint_pos=pad(motion.joint_pos),
    fps=motion.fps,
  )


def support_slip(
  kin: FootKinematics, single: np.ndarray, support: np.ndarray, fps: float
) -> float:
  """Mean horizontal speed of the support foot in single support (m/s)."""
  foot_xy = kin.ankle_pos[np.arange(len(support)), support, :2]
  same = single[1:] & single[:-1] & (support[1:] == support[:-1])
  if not same.any():
    return 0.0
  speed = np.linalg.norm(np.diff(foot_xy, axis=0), axis=1) * fps
  return float(speed[same].mean())


def com_deviation(
  kin: FootKinematics, single: np.ndarray, support: np.ndarray
) -> np.ndarray:
  """Horizontal distance from the COM to the support sole center.

  Double-support frames are NaN.
  """
  foot_xy = kin.foot_pos[np.arange(len(support)), support, :2]
  dev = np.linalg.norm(kin.com_pos[:, :2] - foot_xy, axis=1)
  return np.where(single, dev, np.nan)


@dataclass(frozen=True)
class RefineReport:
  frames_in: int
  frames_out: int
  single_support_frames: int
  slip_before: float
  """Support foot slip in single support before refinement (m/s)."""
  slip_after: float
  """Support foot slip in single support after refinement (m/s)."""
  com_max_deviation: float
  """Largest COM to support foot distance in single support (m)."""
  com_frames_over: int
  """Single-support frames where the COM is beyond ``com_threshold``."""
  padding: tuple[int, int]
  height_shift: float
  """Median vertical shift of the frames by grounding (m)."""
  tilt_before: float
  """Median tilt of the grounded feet before refinement (degrees)."""
  tilt_after: float
  """Median tilt of the grounded feet after refinement (degrees)."""
  warnings: tuple[str, ...]


def _grounded_tilt(kin: FootKinematics, contact_height_threshold: float) -> np.ndarray:
  """Tilts (degrees) of the grounded feet over all frames, flattened."""
  grounded = grounded_feet(kin.ankle_pos[..., 2], contact_height_threshold)
  return foot_tilt(kin)[grounded]


def refine_motion(
  motion: Motion,
  kinematics: G1Kinematics,
  contact_height_threshold: float = 0.05,
  com_threshold: float = 0.2,
  level: bool = True,
  level_height_range: tuple[float, float] = (0.03, 0.10),
  ground: bool = True,
  recenter: bool = True,
  pad: bool = True,
) -> tuple[Motion, RefineReport]:
  """Apply foot leveling and the HuB refinement steps to one clip."""
  kin = kinematics(motion)
  tilt_before = _grounded_tilt(kin, contact_height_threshold)
  if level:
    motion = level_feet(motion, kinematics, kin, level_height_range)
    kin = kinematics(motion)
  single, support = support_labels(kin.ankle_pos[..., 2], contact_height_threshold)

  offsets = grounded_foot_offsets(kin.ankle_pos, single, support)
  height_shift = 0.0
  if ground:
    # Per frame rather than one shift for the clip: leveling changes the sole
    # heights frame by frame (e.g. heels put down in a squat), which a single
    # shift would leave floating or sunk.
    shift = -kin.sole_height.min(axis=1)
    offsets[:, 2] += shift
    height_shift = float(np.median(shift))
  if recenter:
    offsets[:, :2] -= motion.root_pos[0, :2] + offsets[0, :2]
  refined = Motion(
    root_pos=motion.root_pos + offsets,
    root_quat=motion.root_quat,
    joint_pos=motion.joint_pos,
    fps=motion.fps,
  )

  warnings = []
  padding = (0, 0)
  if pad:
    if single[0] or single[-1]:
      warnings.append("clip starts or ends in single support")
    padding = transition_padding(refined, single)
    refined = pad_motion(refined, *padding)

  # Measure slip on fresh FK of the motion that is written.
  refined_kin = kinematics(refined)
  refined_single, refined_support = support_labels(
    refined_kin.ankle_pos[..., 2], contact_height_threshold
  )
  tilt_after = _grounded_tilt(refined_kin, contact_height_threshold)
  if level and tilt_after.max() > 5.0:
    warnings.append(
      f"a grounded foot is still tilted by {tilt_after.max():.1f} deg "
      "(ankle joint limit)"
    )
  dev = com_deviation(kin, single, support)
  report = RefineReport(
    frames_in=motion.num_frames,
    frames_out=refined.num_frames,
    single_support_frames=int(single.sum()),
    slip_before=support_slip(kin, single, support, motion.fps),
    slip_after=support_slip(refined_kin, refined_single, refined_support, motion.fps),
    com_max_deviation=float(np.nanmax(dev)) if single.any() else 0.0,
    com_frames_over=int(np.sum(dev > com_threshold)),
    padding=padding,
    height_shift=height_shift,
    tilt_before=float(np.median(tilt_before)),
    tilt_after=float(np.median(tilt_after)),
    warnings=tuple(warnings),
  )
  return refined, report


##
# CLI.
##


def main(
  input_path: str,
  output_dir: str = "motions/hub_refined",
  input_fps: float | None = None,
  contact_height_threshold: float = 0.05,
  com_threshold: float = 0.2,
  keep_rejected: bool = False,
  level: bool = True,
  level_height_range: tuple[float, float] = (0.03, 0.10),
  ground: bool = True,
  recenter: bool = True,
  pad: bool = True,
  npz: bool = True,
  device: str = "cuda:0",
) -> None:
  """Refine reference motions and write one CSV (and npz) per clip.

  Args:
    input_path: A HuB/PHC motion pickle (.pkl), an mjlab motion CSV (.csv), or a
      directory of them.
    output_dir: Directory for the refined clips.
    input_fps: Frame rate of CSV inputs. Pickles store their own.
    contact_height_threshold: Ankle height difference that marks single support.
    com_threshold: Largest allowed distance between the ground-projected COM and
      the support foot in single support. Clips beyond it are rejected.
    keep_rejected: Write clips that fail the COM check anyway.
    level: Level the soles of feet on or near the ground.
    level_height_range: Sole heights (m) where foot leveling starts to fade and
      where it is gone.
    ground: Shift each frame vertically so its lower sole touches the ground.
    recenter: Shift each clip horizontally so it starts at the origin.
    pad: Repeat the first and last frames for transition stabilization.
    npz: Convert each written CSV to npz with ``mjlab.scripts.csv_to_npz``.
    device: Device for the npz conversion.
  """
  paths = [Path(input_path)]
  if paths[0].is_dir():
    paths = sorted(p for p in paths[0].iterdir() if p.suffix in (".pkl", ".csv"))
  clips: dict[str, Motion] = {}
  for path in paths:
    if path.suffix == ".pkl":
      clips.update(load_motion_pkl(path))
    elif path.suffix == ".csv":
      if input_fps is None:
        raise ValueError("--input-fps is required for CSV inputs")
      clips[path.stem] = load_motion_csv(path, input_fps)
    else:
      raise ValueError(f"Unsupported motion file: {path}")

  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  kinematics = G1Kinematics()
  written: list[tuple[Path, float]] = []
  print(
    f"{'clip':34s} {'frames':>9s} {'single':>6s} {'slip mm/s':>11s} "
    f"{'tilt deg':>10s} {'COM max m':>11s} {'padding':>9s} {'dz m':>7s}  status"
  )
  for name, motion in clips.items():
    refined, r = refine_motion(
      motion,
      kinematics,
      contact_height_threshold=contact_height_threshold,
      com_threshold=com_threshold,
      level=level,
      level_height_range=level_height_range,
      ground=ground,
      recenter=recenter,
      pad=pad,
    )
    rejected = r.com_frames_over > 0
    status = "rejected (COM)" if rejected else "ok"
    if rejected and keep_rejected:
      status = "kept (COM)"
    print(
      f"{name[:34]:34s} {f'{r.frames_in}->{r.frames_out}':>9s} "
      f"{r.single_support_frames:>6d} "
      f"{f'{r.slip_before * 1e3:.1f}->{r.slip_after * 1e3:.1f}':>11s} "
      f"{f'{r.tilt_before:.1f}->{r.tilt_after:.1f}':>10s} "
      f"{f'{r.com_max_deviation:.3f}({r.com_frames_over})':>11s} "
      f"{f'+{r.padding[0]}/+{r.padding[1]}':>9s} {r.height_shift:>7.3f}  {status}"
    )
    for warning in r.warnings:
      print(f"  warning: {warning}")
    if rejected and not keep_rejected:
      continue
    csv_path = out_dir / f"{name}.csv"
    save_motion_csv(refined, csv_path)
    written.append((csv_path, refined.fps))

  if npz:
    for csv_path, fps in written:
      csv_to_npz.main(
        input_file=str(csv_path),
        output_file=str(csv_path.with_suffix(".npz")),
        input_fps=fps,
        device=device,
      )
  print(f"\nWrote {len(written)} of {len(clips)} clips to {out_dir}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
