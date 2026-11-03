"""Play a policy while it is still training, always following the latest checkpoint.

It finds the currently-training run under ``logs/rsl_rl/<experiment>/`` (the one
with the most recently written ``model_*.pt``), loads its newest checkpoint, and
plays it in the viewer. Every ``--reload-interval-s`` seconds it polls the run
directory and hot-swaps to a newer checkpoint as soon as training saves one, so
you can watch the policy improve live.

For tracking tasks the reference motion is read from the run's
``params/env.yaml`` (the exact npz the run trained on). Override it with
``--motion-file``.

Usage:
  uv run python play_latest.py Mjlab-HuB-Teacher-Unitree-G1 --viewer viser --device cpu
  uv run python play_latest.py Mjlab-HuB-Teacher-Unitree-G1 \\
    --run-dir logs/rsl_rl/g1_hub_teacher/2026-11-03_13-27-26
  # Follow whichever run of an experiment is training, swapping the motion when a
  # new run (e.g. the next clip of a batch) starts.
  uv run python play_latest.py Mjlab-HuB-Teacher-Unitree-G1 \\
    --experiment logs/rsl_rl/g1_hub_teacher --viewer viser
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionLoader
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def _checkpoint_step(path: Path) -> int:
  nums = re.findall(r"\d+", path.stem)
  return int(nums[-1]) if nums else -1


def _latest_checkpoint(run_dir: Path) -> Path | None:
  candidates = [p for p in run_dir.glob("model_*.pt") if p.is_file()]
  if not candidates:
    return None
  # Prefer the highest training step; break ties by modification time.
  return max(candidates, key=lambda p: (_checkpoint_step(p), p.stat().st_mtime))


def _find_active_run_dir(experiment_dir: Path) -> Path:
  """Return the run dir whose newest checkpoint was modified most recently."""
  best: tuple[float, Path] | None = None
  for run_dir in experiment_dir.iterdir():
    if not run_dir.is_dir():
      continue
    ckpt = _latest_checkpoint(run_dir)
    if ckpt is None:
      continue
    mtime = ckpt.stat().st_mtime
    if best is None or mtime > best[0]:
      best = (mtime, run_dir)
  if best is None:
    raise FileNotFoundError(
      f"No checkpoints found under {experiment_dir}. Is training running?"
    )
  return best[1]


def _resolve_motion_file(run_dir: Path) -> str:
  """Read the reference motion path from the run's dumped env config."""
  env_yaml = run_dir / "params" / "env.yaml"
  if not env_yaml.exists():
    raise FileNotFoundError(f"{env_yaml} not found; pass --motion-file explicitly.")
  # Line parsing rather than a YAML parser: dump_yaml emits Python-specific tags
  # (e.g. !!python/tuple) that safe_load rejects, and only this path is needed.
  for line in env_yaml.read_text().splitlines():
    match = re.match(r"\s*motion_file:\s*(.+?)\s*$", line)
    if match:
      value = match.group(1).strip("'\"")
      if value:
        return value
  raise ValueError(f"No motion_file found in {env_yaml}; pass --motion-file.")


class LatestCheckpointPolicy:
  """Callable policy that hot-reloads the newest checkpoint of the run it follows.

  The viewer calls ``policy(obs)`` every control step; the directory poll is
  throttled to once per ``interval_s`` and a checkpoint is only reloaded when a
  newer one appears. A checkpoint still being written fails to load and is
  retried on the next poll.

  If ``experiment_dir`` is given, it also follows whichever run of that
  experiment is active (newest checkpoint). When the active run changes, the
  reference motion is swapped in place. All runs must share the task's
  observation and action layout.
  """

  def __init__(
    self,
    runner: MjlabOnPolicyRunner,
    env: RslRlVecEnvWrapper,
    run_dir: Path,
    device: str,
    interval_s: float = 2.0,
    experiment_dir: Path | None = None,
    motion_file: str | None = None,
  ) -> None:
    self.runner = runner
    self.env = env
    self.run_dir = run_dir
    self.device = device
    self.interval_s = interval_s
    self.experiment_dir = experiment_dir
    self._motion_file = motion_file
    self._policy = None
    self._loaded_path: Path | None = None
    self._loaded_mtime = -1.0
    self._last_check = 0.0
    self._reload(force=True)
    if self._policy is None:
      raise FileNotFoundError(f"No loadable checkpoint in {run_dir}")

  def _swap_motion(self, motion_file: str) -> None:
    command = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
    command.motion = MotionLoader(motion_file, command.body_indexes, command.device)
    command.time_steps.zero_()
    self.env.reset()
    self._motion_file = motion_file
    print(f"[play_latest] motion -> {motion_file}", flush=True)

  def _maybe_switch_run(self) -> None:
    if self.experiment_dir is None:
      return
    try:
      active = _find_active_run_dir(self.experiment_dir)
    except FileNotFoundError:
      return
    if active == self.run_dir:
      return
    print(f"[play_latest] active run -> {active.name}", flush=True)
    self.run_dir = active
    self._loaded_path = None  # Force a checkpoint reload from the new run.
    if self._motion_file is None:
      return  # Not a tracking task.
    try:
      motion_file = _resolve_motion_file(active)
    except (FileNotFoundError, ValueError):
      return  # params/env.yaml not written yet; retry on the next poll.
    if motion_file != self._motion_file and Path(motion_file).exists():
      self._swap_motion(motion_file)

  def _reload(self, force: bool = False) -> None:
    now = time.perf_counter()
    if not force and now - self._last_check < self.interval_s:
      return
    self._last_check = now

    self._maybe_switch_run()

    latest = _latest_checkpoint(self.run_dir)
    if latest is None:
      return
    mtime = latest.stat().st_mtime
    if latest == self._loaded_path and mtime == self._loaded_mtime:
      return

    try:
      self.runner.load(
        str(latest), load_cfg={"actor": True}, strict=True, map_location=self.device
      )
      self._policy = self.runner.get_inference_policy(device=self.device)
    except Exception as exc:  # noqa: BLE001 - file may be mid-write; retry later.
      print(f"[play_latest] skip {latest.name} (still writing?): {type(exc).__name__}")
      return

    self._loaded_path = latest
    self._loaded_mtime = mtime
    print(f"[play_latest] now playing: {self.run_dir.name}/{latest.name}", flush=True)

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    self._reload()
    assert self._policy is not None
    return self._policy(obs)

  def reset(self, dones: torch.Tensor | None = None) -> None:
    """Called by the viewer after every step with the done flags."""
    del dones  # The MLP policy keeps no state.


@dataclass(frozen=True)
class Config:
  run_dir: str | None = None
  """A single run directory to follow. Defaults to the currently-training run."""
  experiment: str | None = None
  """Experiment dir (e.g. logs/rsl_rl/g1_hub_teacher) to follow: tracks whichever
  run is training and switches (swapping the motion) when a new run starts.
  Takes precedence over --run-dir."""
  log_root: str = "logs/rsl_rl"
  """Root directory of the experiment logs, used when neither is given."""
  motion_file: str | None = None
  """Reference motion npz. Defaults to the run's params/env.yaml value."""
  reload_interval_s: float = 2.0
  """How often (seconds) to poll for a newer checkpoint or run."""
  num_envs: int | None = None
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  episode_length_s: float | None = None
  """Reset each env after this many seconds. Play mode defaults to effectively
  infinite (each clip plays start to end)."""
  sampling_mode: Literal["start", "uniform"] | None = None
  """Motion start frame on reset. Play mode defaults to 'start'; 'uniform' picks a
  random frame. Combine with --episode-length-s 10 to play random windows."""


def run(task_id: str, cfg: Config) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  experiment_dir: Path | None = None
  if cfg.experiment is not None:
    experiment_dir = Path(cfg.experiment).resolve()
    run_dir = _find_active_run_dir(experiment_dir)
    print(
      f"[play_latest] following experiment: {experiment_dir} (active: {run_dir.name})"
    )
  elif cfg.run_dir is not None:
    run_dir = Path(cfg.run_dir).resolve()
    print(f"[play_latest] following run: {run_dir}")
  else:
    run_dir = _find_active_run_dir(
      (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    )
    print(f"[play_latest] following run: {run_dir}")

  # Tracking tasks need the reference motion of the run; others do not.
  motion_file: str | None = None
  motion_cmd = env_cfg.commands.get("motion")
  if isinstance(motion_cmd, MotionCommandCfg):
    motion_file = cfg.motion_file or _resolve_motion_file(run_dir)
    if not Path(motion_file).exists():
      raise FileNotFoundError(f"Motion file does not exist: {motion_file}")
    motion_cmd.motion_file = motion_file
    if cfg.sampling_mode is not None:
      motion_cmd.sampling_mode = cfg.sampling_mode
      print(f"[play_latest] sampling_mode: {cfg.sampling_mode}")
    print(f"[play_latest] motion: {motion_file}")

  if cfg.episode_length_s is not None:
    env_cfg.episode_length_s = cfg.episode_length_s
    print(f"[play_latest] episode_length_s: {cfg.episode_length_s}")

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs

  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None),
    clip_actions=agent_cfg.clip_actions,
  )
  env.reset()

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)

  policy = LatestCheckpointPolicy(
    runner,
    env,
    run_dir,
    device,
    interval_s=cfg.reload_interval_s,
    experiment_dir=experiment_dir,
    motion_file=motion_file,
  )

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = cfg.viewer

  print(f"[play_latest] viewer: {resolved} | polling every {cfg.reload_interval_s}s")
  if resolved == "native":
    NativeMujocoViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy).run()

  env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401  (populate the task registry)

  all_tasks = list_tasks()
  chosen_task, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    Config,
    args=remaining,
    default=Config(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run(chosen_task, cfg)


if __name__ == "__main__":
  main()
