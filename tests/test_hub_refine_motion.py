"""Tests for the HuB reference motion refinement script."""

import collections
import io
import pickle
import sys
import types

import numpy as np
import pytest

from mjlab.tasks.hub.scripts.refine_motion import (
  G1_JOINT_NAMES,
  G1Kinematics,
  Motion,
  foot_tilt,
  load_motion_pkl,
  refine_motion,
  support_labels,
)


class _JoblibStylePickler(pickle._Pickler):
  """Writes arrays like joblib: a pickled wrapper followed by the raw bytes."""

  def __init__(self, file, wrapper_cls) -> None:
    super().__init__(file, protocol=4)
    self._file = file
    self._wrapper_cls = wrapper_cls

  # The pure-Python pickler internals used below are missing from typeshed.
  def save(self, obj, save_persistent_id=True):
    if not isinstance(obj, np.ndarray):
      return super().save(obj, save_persistent_id)  # type: ignore[unresolved-attribute]
    wrapper = self._wrapper_cls()
    wrapper.__dict__.update(
      subclass=np.ndarray,
      shape=obj.shape,
      order="C",
      dtype=obj.dtype,
      allow_mmap=False,
      numpy_array_alignment_bytes=16,
    )
    super().save(wrapper)  # type: ignore[unresolved-attribute]
    self.framer.commit_frame(force=True)  # type: ignore[unresolved-attribute]
    padding = 16 - (self._file.tell() + 1) % 16
    self._file.write(bytes([padding]) + b"\xff" * padding)
    self._file.write(obj.tobytes())


def test_load_motion_pkl_reads_joblib_and_rejects_other_globals(
  tmp_path, monkeypatch
) -> None:
  module = types.ModuleType("joblib.numpy_pickle")
  wrapper_cls = type("NumpyArrayWrapper", (), {"__module__": module.__name__})
  module.NumpyArrayWrapper = wrapper_cls  # type: ignore[attr-defined]
  # Stub joblib so pickle can resolve the wrapper class by name.
  monkeypatch.setitem(sys.modules, "joblib", types.ModuleType("joblib"))
  monkeypatch.setitem(sys.modules, module.__name__, module)

  rng = np.random.default_rng(0)
  root_rot = rng.normal(size=(7, 4))
  clip = {
    "root_trans_offset": rng.normal(size=(7, 3)),
    "root_rot": root_rot,
    "dof": rng.normal(size=(7, 29)).astype(np.float32),
    "fps": 30,
  }
  buf = io.BytesIO()
  _JoblibStylePickler(buf, wrapper_cls).dump({"clip": clip})
  path = tmp_path / "motion.pkl"
  path.write_bytes(buf.getvalue())

  motion = load_motion_pkl(path)["clip"]
  np.testing.assert_array_equal(motion.root_pos, clip["root_trans_offset"])
  np.testing.assert_array_equal(motion.root_quat, root_rot[:, [3, 0, 1, 2]])
  np.testing.assert_array_equal(motion.joint_pos, clip["dof"])
  assert motion.fps == 30.0

  path.write_bytes(pickle.dumps({"clip": collections.OrderedDict()}))
  with pytest.raises(pickle.UnpicklingError, match="collections.OrderedDict"):
    load_motion_pkl(path)


def test_refine_motion_pins_support_foot_grounds_and_pads() -> None:
  # Stand for 5 frames, lift the right leg for 30 frames while the whole body
  # slides forward, then stand for 5 frames.
  num_frames = 40
  joint_pos = np.zeros((num_frames, len(G1_JOINT_NAMES)))
  lifted = slice(5, 35)
  joint_pos[lifted, G1_JOINT_NAMES.index("right_hip_pitch_joint")] = -0.8
  joint_pos[lifted, G1_JOINT_NAMES.index("right_knee_joint")] = 1.2
  root_pos = np.tile([1.0, 2.0, 0.85], (num_frames, 1))
  root_pos[lifted, 0] += 0.003 * np.arange(30)
  root_pos[35:, 0] = root_pos[34, 0]
  root_quat = np.tile([1.0, 0.0, 0.0, 0.0], (num_frames, 1))
  motion = Motion(root_pos, root_quat, joint_pos, fps=30.0)

  kinematics = G1Kinematics()
  refined, report = refine_motion(motion, kinematics)

  assert report.single_support_frames == 30
  assert report.slip_before > 0.05
  assert report.slip_after < 1e-9
  # 30 balance frames need 15 copies per side; 4 are already at each end.
  assert report.padding == (11, 11)
  assert refined.num_frames == num_frames + 22

  kin = kinematics(refined)
  single, support = support_labels(kin.ankle_pos[..., 2])
  assert single.sum() == 30 and np.all(support[single] == 0)
  left_ankle_xy = kin.ankle_pos[single, 0, :2]
  np.testing.assert_allclose(left_ankle_xy, left_ankle_xy[:1].repeat(30, 0))
  np.testing.assert_allclose(kin.sole_height.min(axis=1), 0.0, atol=1e-9)
  np.testing.assert_allclose(refined.root_pos[0, :2], 0.0, atol=1e-12)


def test_refine_motion_levels_feet_on_the_ground_only() -> None:
  # Stand on pitched and rolled feet for 10 frames, then lift the right leg
  # high with the same ankle angles for 10 frames.
  num_frames = 20
  joint_pos = np.zeros((num_frames, len(G1_JOINT_NAMES)))
  right_ankle = [
    G1_JOINT_NAMES.index(f"right_ankle_{name}_joint") for name in ("pitch", "roll")
  ]
  for side in ("left", "right"):
    joint_pos[:, G1_JOINT_NAMES.index(f"{side}_ankle_pitch_joint")] = 0.2
    joint_pos[:, G1_JOINT_NAMES.index(f"{side}_ankle_roll_joint")] = 0.15
  joint_pos[10:, G1_JOINT_NAMES.index("right_hip_pitch_joint")] = -1.2
  joint_pos[10:, G1_JOINT_NAMES.index("right_knee_joint")] = 1.5
  root_pos = np.tile([0.0, 0.0, 0.85], (num_frames, 1))
  root_quat = np.tile([1.0, 0.0, 0.0, 0.0], (num_frames, 1))
  motion = Motion(root_pos, root_quat, joint_pos, fps=30.0)

  kinematics = G1Kinematics()
  refined, report = refine_motion(motion, kinematics, pad=False)

  assert report.tilt_before > 10.0
  assert report.tilt_after < 0.1
  kin = kinematics(refined)
  tilt = foot_tilt(kin)
  np.testing.assert_allclose(tilt[:10], 0.0, atol=0.1)  # Both standing feet.
  np.testing.assert_allclose(tilt[10:, 0], 0.0, atol=0.1)  # The support foot.
  # The lifted foot keeps its reference orientation.
  np.testing.assert_array_equal(
    refined.joint_pos[10:, right_ankle], motion.joint_pos[10:, right_ankle]
  )
  np.testing.assert_allclose(kin.sole_height.min(axis=1), 0.0, atol=1e-9)
