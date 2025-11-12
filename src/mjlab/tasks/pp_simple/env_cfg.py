"""Pick-and-place simple task with a staged reward curriculum.

This configuration replaces the legacy `env_cfg_og.py` setup. It keeps the same
scene, observations, actions, events, rewards, and terminations while adding a
three-stage curriculum that progressively enables reward terms:

    Stage 0 → reach the object and open the gripper
    Stage 1 → touch the object and close the gripper (Stage 0 stays active)
    Stage 2 → lift the object while maintaining prior-stage rewards
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple, cast

import torch  # pyright: ignore[reportMissingImports]

from mjlab.asset_zoo.misc_assets.blue_cylinder.blue_cylinder_constants import (
    BLUE_CYLINDER_CFG,
)
from mjlab.asset_zoo.robots.SO_101.so101_constants import SO101_ROBOT_CFG
from mjlab.envs import ManagerBasedEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions.actions_config import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.manager_term_config import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewardTerm,
    TerminationTermCfg as DoneTerm,
    term,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


__all__ = [
    "SCENE_CFG",
    "SIM_CFG",
    "ObservationCfg",
    "ActionCfg",
    "RewardCfg",
    "TerminationCfg",
    "EventCfg",
    "CurriculumCfg",
    "PPSimpleEnvCfg",
    "PPSimpleEnvCfgPlay",
]


# -----------------------------------------------------------------------------
# Scene configuration
# -----------------------------------------------------------------------------

SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(terrain_type="plane", env_spacing=1.0),
    num_envs=1024,
    entities={
        "robot": SO101_ROBOT_CFG,
        "object": BLUE_CYLINDER_CFG,
    },
)


# -----------------------------------------------------------------------------
# Helper accessors
# -----------------------------------------------------------------------------


def _robot_body_index(env: ManagerBasedEnv, body_name: str) -> int:
    cache_name = f"_robot_body_id_{body_name}"
    if not hasattr(env, cache_name):
        ids, _ = env.scene["robot"].find_bodies(body_name, preserve_order=True)
        setattr(env, cache_name, int(ids[0]))
    return getattr(env, cache_name)


def _robot_geom_index(env: ManagerBasedEnv, geom_name: str) -> int:
    cache_name = f"_robot_geom_id_{geom_name}"
    if not hasattr(env, cache_name):
        ids, _ = env.scene["robot"].find_geoms(geom_name, preserve_order=True)
        setattr(env, cache_name, int(ids[0]))
    return getattr(env, cache_name)


def _object_body_index(env: ManagerBasedEnv, body_name: str) -> int:
    cache_name = f"_object_body_id_{body_name}"
    if not hasattr(env, cache_name):
        ids, _ = env.scene["object"].find_bodies(body_name, preserve_order=True)
        setattr(env, cache_name, int(ids[0]))
    return getattr(env, cache_name)


def _object_geom_index(env: ManagerBasedEnv, geom_name: str) -> int:
    cache_name = f"_object_geom_id_{geom_name}"
    if not hasattr(env, cache_name):
        ids, _ = env.scene["object"].find_geoms(geom_name, preserve_order=True)
        setattr(env, cache_name, int(ids[0]))
    return getattr(env, cache_name)


def _object_collision_pos(env: ManagerBasedEnv) -> torch.Tensor:
    geom_id = _object_geom_index(env, "blue_cylinder_collision")
    return env.scene["object"].data.geom_pos_w[:, geom_id, :]


_OBJECT_RADIUS = 0.0175
_OBJECT_HALF_HEIGHT = 0.035 / 2.0
_GRASP_MIN_FORCE = 0.05  # Much lower threshold for success


def _gripper_contact_forces(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contact force magnitudes on each fingertip body."""
    robot = env.scene["robot"]
    device = robot.data.body_link_pos_w.device

    cfrc_ext = getattr(env.sim.data, "cfrc_ext", None)
    if cfrc_ext is None:
        zeros = torch.zeros(env.num_envs, device=device)
        return zeros, zeros

    body_ids_global = robot.data.indexing.body_ids
    tp1_global = int(body_ids_global[_robot_body_index(env, "touch_point1")])
    tp2_global = int(body_ids_global[_robot_body_index(env, "touch_point2")])

    tp1_force_vec = cfrc_ext[:, tp1_global, :3]
    tp2_force_vec = cfrc_ext[:, tp2_global, :3]

    return torch.norm(tp1_force_vec, dim=-1), torch.norm(tp2_force_vec, dim=-1)

def _touch_point1_pos(env: ManagerBasedEnv) -> torch.Tensor:
    body_id = _robot_body_index(env, "touch_point1")
    robot = env.scene["robot"]
    return robot.data.body_link_pos_w[:, body_id, :]


def _touch_point2_pos(env: ManagerBasedEnv) -> torch.Tensor:
    body_id = _robot_body_index(env, "touch_point2")
    robot = env.scene["robot"]
    return robot.data.body_link_pos_w[:, body_id, :]


def _gripper_mid_pos(env: ManagerBasedEnv) -> torch.Tensor:
    geom_id = _robot_geom_index(env, "gripper_mid_point")
    robot = env.scene["robot"]
    return robot.data.geom_pos_w[:, geom_id, :]


def _object_surface1_pos(env: ManagerBasedEnv) -> torch.Tensor:
    object_pos = _object_collision_pos(env)
    offset = object_pos.new_tensor([0.0, -_OBJECT_RADIUS, _OBJECT_HALF_HEIGHT])
    return object_pos + offset


def _object_surface2_pos(env: ManagerBasedEnv) -> torch.Tensor:
    object_pos = _object_collision_pos(env)
    offset = object_pos.new_tensor([0.0, _OBJECT_RADIUS, _OBJECT_HALF_HEIGHT])
    return object_pos + offset


def _object_middle_pos(env: ManagerBasedEnv) -> torch.Tensor:
    object_pos = _object_collision_pos(env)
    offset = object_pos.new_tensor([0.0, 0.0, _OBJECT_HALF_HEIGHT])
    return object_pos + offset


def _touch_distances(env: ManagerBasedEnv) -> Tuple[torch.Tensor, torch.Tensor]:
    middle = _object_middle_pos(env)
    delta1 = middle - _touch_point1_pos(env)
    delta2 = middle - _touch_point2_pos(env)
    return torch.norm(delta1, dim=-1), torch.norm(delta2, dim=-1)


def _gripper_gap(env: ManagerBasedEnv) -> torch.Tensor:
    return torch.norm(_touch_point1_pos(env) - _touch_point2_pos(env), dim=-1)


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------


def _obs_object_to_gripper(env: ManagerBasedEnv) -> torch.Tensor:
    return _object_middle_pos(env) - _gripper_mid_pos(env)


@dataclass
class ObservationCfg:
    """Observation terms exposed to both policy and critic."""

    @dataclass
    class PolicyCfg(ObsGroup):
        joint_pos: ObsTerm = term(ObsTerm, func=mdp.joint_pos_rel)
        object_to_gripper: ObsTerm = term(ObsTerm, func=_obs_object_to_gripper)

    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: PolicyCfg = field(default_factory=PolicyCfg)


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------


@dataclass
class ActionCfg:
    joint_pos: JointPositionActionCfg = term(
        JointPositionActionCfg,
        asset_name="robot",
        actuator_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------

_GRIPPER_APPROACH_THRESHOLD = 0.008
_TOUCH_THRESHOLD = 0.001


def _reward_reach(env: ManagerBasedEnv) -> torch.Tensor:
    object_collision = _object_collision_pos(env)
    offset_z = object_collision.new_tensor([0.0, 0.0, _OBJECT_HALF_HEIGHT])
    offset_y = object_collision.new_tensor([0.0, _OBJECT_RADIUS, 0.0])
    obj_middle_pos = object_collision + offset_z
    object_surface1_pos = object_collision + offset_z - offset_y
    object_surface2_pos = object_collision + offset_z + offset_y
    gripper_pos = _gripper_mid_pos(env)
    touch_point1_pos = _touch_point1_pos(env)
    touch_point2_pos = _touch_point2_pos(env)

    delta = obj_middle_pos - gripper_pos
    deltatp1 = object_surface1_pos - touch_point1_pos
    deltatp2 = object_surface2_pos - touch_point2_pos

    distance_tp = torch.norm(deltatp1 + deltatp2, dim=-1)

    xy_weight = 50.0
    z_weight = 15.0

    xy_distance = torch.norm(delta[:, :2], dim=-1)
    z_distance = torch.abs(delta[:, 2])
    total_distance = torch.norm(delta, dim=-1)

    xy_reward = torch.exp(-xy_weight * xy_distance)
    z_reward = torch.exp(-z_weight * z_distance)
    tp_reward = torch.exp(-10.0 * distance_tp)
    sharp_bonus = torch.exp(-20.0 * total_distance)
    distance_reward = xy_reward * z_reward + 0.75 * sharp_bonus + tp_reward

    geom_id = _robot_geom_index(env, "gripper_mid_point")
    robot = env.scene["robot"]
    gripper_vel = robot.data.geom_lin_vel_w[:, geom_id, :]

    xy_unit = delta[:, :2] / (xy_distance.unsqueeze(-1) + 1e-6)
    xy_vel_toward = torch.sum(gripper_vel[:, :2] * xy_unit, dim=-1)
    z_vel_toward = gripper_vel[:, 2] * torch.sign(delta[:, 2])

    near_mask = (total_distance < 0.05).float()
    velocity_components = 0.1 * torch.clamp(xy_vel_toward, min=0.0) + 0.02 * torch.clamp(
        z_vel_toward, min=0.0
    )
    velocity_bonus = near_mask * torch.clamp(velocity_components, max=0.25)

    raw_reward = distance_reward + velocity_bonus
    return torch.clamp(raw_reward / 3.0, 0.0, 1.0)


def _reward_touch(env: ManagerBasedEnv) -> torch.Tensor:
    object_surface1_pos = _object_surface1_pos(env)
    object_surface2_pos = _object_surface2_pos(env)
    touch_point1_pos = _touch_point1_pos(env)
    touch_point2_pos = _touch_point2_pos(env)

    delta1 = object_surface1_pos - touch_point1_pos
    delta2 = object_surface2_pos - touch_point2_pos

    distance1 = torch.norm(delta1, dim=-1)
    distance2 = torch.norm(delta2, dim=-1)

    threshold = _GRIPPER_APPROACH_THRESHOLD

    normalized1 = torch.clamp(distance1 / threshold, min=0.0, max=1.0)
    normalized2 = torch.clamp(distance2 / threshold, min=0.0, max=1.0)

    touch_reward1 = torch.where(
        distance1 <= threshold,
        torch.exp(-500.0 * normalized1) + 20.0,
        torch.zeros_like(distance1),
    )
    touch_reward2 = torch.where(
        distance2 <= threshold,
        torch.exp(-500.0 * normalized2) + 100.0,
        torch.zeros_like(distance2),
    )

    touch_reward = touch_reward1 + touch_reward2
    return torch.clamp(touch_reward, 0.0, 1.0)


def _reward_open_gripper(env: ManagerBasedEnv) -> torch.Tensor:
    rl_env = cast("ManagerBasedRlEnv", env)
    gap = _gripper_gap(env)

    target_gap = 0.035
    max_reward = 1.0

    base_reward = torch.clamp((gap - target_gap) / target_gap, min=0.0, max=max_reward)

    if (not hasattr(rl_env, "_prev_open_gap")) or rl_env._prev_open_gap.shape[0] != rl_env.num_envs:
        rl_env._prev_open_gap = gap.clone()
    else:
        if hasattr(rl_env, "reset_buf"):
            reset_buf = getattr(rl_env, "reset_buf")
            if isinstance(reset_buf, torch.Tensor) and reset_buf.any():
                rl_env._prev_open_gap = rl_env._prev_open_gap.clone()
                rl_env._prev_open_gap[reset_buf.bool()] = gap[reset_buf.bool()]

    gap_delta = torch.clamp(gap - rl_env._prev_open_gap, min=0.0)
    rl_env._prev_open_gap = gap.clone()

    exploration_bonus = torch.clamp(gap_delta * 5.0, max=0.5)

    dist1, dist2 = _touch_distances(env)
    touching = dist1 <= _TOUCH_THRESHOLD

    raw_reward = torch.where(
        touching,
        torch.full_like(base_reward, max_reward),
        base_reward + exploration_bonus,
    )

    return torch.clamp(raw_reward / max_reward, 0.0, 1.0)


def _reward_close_gripper(env: ManagerBasedEnv) -> torch.Tensor:
    rl_env = cast("ManagerBasedRlEnv", env)
    dist1, dist2 = _touch_distances(env)
    touching = (dist1 <= _TOUCH_THRESHOLD) & (dist2 <= _TOUCH_THRESHOLD)

    if (not hasattr(rl_env, "_prev_close_dist1")) or rl_env._prev_close_dist1.shape[0] != rl_env.num_envs:
        rl_env._prev_close_dist1 = dist1.clone()
        rl_env._prev_close_dist2 = dist2.clone()
    else:
        if hasattr(rl_env, "reset_buf"):
            reset_mask = getattr(rl_env, "reset_buf")
            if isinstance(reset_mask, torch.Tensor) and reset_mask.any():
                rl_env._prev_close_dist1 = rl_env._prev_close_dist1.clone()
                rl_env._prev_close_dist2 = rl_env._prev_close_dist2.clone()
                rl_env._prev_close_dist1[reset_mask.bool()] = dist1[reset_mask.bool()]
                rl_env._prev_close_dist2[reset_mask.bool()] = dist2[reset_mask.bool()]

    dist1_delta = torch.clamp(rl_env._prev_close_dist1 - dist1, min=0.0)
    dist2_delta = torch.clamp(rl_env._prev_close_dist2 - dist2, min=0.0)

    rl_env._prev_close_dist1 = dist1.clone()
    rl_env._prev_close_dist2 = dist2.clone()

    progress_reward = (dist1_delta + dist2_delta) / (2.0 * _TOUCH_THRESHOLD)
    progress_reward = torch.clamp(progress_reward, 0.0, 1.0)

    # When touching, also reward gripper closing (smaller gap = more closed)
    gap = _gripper_gap(env)
    target_gap_when_touching = 0.01  # Small gap when grasping
    gap_reward = torch.exp(-50.0 * torch.clamp(gap - target_gap_when_touching, min=0.0))
    
    return torch.where(
        touching,
        torch.clamp(progress_reward + 0.5 * gap_reward, 0.0, 1.0),
        progress_reward
    )


def _reward_grasp(env: ManagerBasedEnv) -> torch.Tensor:
    """Simplified grasp reward that rewards any force presence."""
    force_tp1, force_tp2 = _gripper_contact_forces(env)
    
    # Reward based on average force (encourages both fingers to contact)
    avg_force = 0.5 * (force_tp1 + force_tp2)
    
    # Scale reward by force magnitude (saturates around 0.2N)
    force_reward = torch.clamp(avg_force / 0.2, 0.0, 1.0)
    
    # Bonus for balanced forces (both fingers contacting)
    both_contacting = (force_tp1 > 0.05) & (force_tp2 > 0.05)
    balance_bonus = torch.exp(-2.0 * torch.abs(force_tp1 - force_tp2) / (avg_force + 0.01))
    
    reward = force_reward * (0.7 + 0.3 * both_contacting.float() * balance_bonus)
    return torch.clamp(reward, 0.0, 1.0)


def _reward_lift(env: ManagerBasedEnv) -> torch.Tensor:
    obj = env.scene["object"]
    body_id = _object_body_index(env, "blue_cylinder")
    obj_height = obj.data.body_link_pos_w[:, body_id, 2]
    base_height = obj_height.new_tensor(obj.cfg.init_state.pos[2])
    lifted = torch.clamp(obj_height - base_height, min=0.0, max=0.05)
    lift_reward = lifted / 0.05
    return torch.clamp(lift_reward, 0.0, 1.0)


def _reward_upreach(env: ManagerBasedEnv) -> torch.Tensor:
    rl_env = cast("ManagerBasedRlEnv", env)
    dist1, dist2 = _touch_distances(env)
    touching = (dist1 <= _TOUCH_THRESHOLD) & (dist2 <= _TOUCH_THRESHOLD)
    touch_mask = touching.float()

    robot = env.scene["robot"]
    geom_ids, _ = robot.find_geoms("gripper_mid_point", preserve_order=True)
    gripper_pos = robot.data.geom_pos_w[:, geom_ids[0], :]
    gripper_height = gripper_pos[:, 2]

    if (
        not hasattr(rl_env, "_initial_gripper_height")
        or rl_env._initial_gripper_height.shape[0] != rl_env.num_envs
    ):
        rl_env._initial_gripper_height = gripper_height.clone()

    height_gain = torch.clamp(
        gripper_height - rl_env._initial_gripper_height, min=0.0, max=0.1
    )
    lift_reward = height_gain / 0.1
    up_reward = touch_mask * torch.clamp(lift_reward, 0.0, 1.0)
    return torch.clamp(up_reward, 0.0, 1.0)


def _cost_wrist_roll(env: ManagerBasedEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos[:, 4]
    target = torch.as_tensor(0, device=joint_pos.device, dtype=joint_pos.dtype)
    deviation = torch.abs(joint_pos - target)
    reward = torch.exp(-100.0 * deviation)
    return -torch.clamp(reward, 0.0, 1.0)


@dataclass
class RewardCfg:
    reach: RewardTerm = term(RewardTerm, func=_reward_reach, weight=5.0)
    touch: RewardTerm = term(RewardTerm, func=_reward_touch, weight=40.0)
    gripper_open: RewardTerm = term(RewardTerm, func=_reward_open_gripper, weight=1.0)
    gripper_close: RewardTerm = term(RewardTerm, func=_reward_close_gripper, weight=50.0)
    grasp: RewardTerm = term(RewardTerm, func=_reward_grasp, weight=15.0)
    lift: RewardTerm = term(RewardTerm, func=_reward_lift, weight=1.0)
    wrist_roll_cost: RewardTerm = term(RewardTerm, func=_cost_wrist_roll, weight=120.0)
    upreach: RewardTerm = term(RewardTerm, func=_reward_upreach, weight=1.0)


# -----------------------------------------------------------------------------
# Terminations and events
# -----------------------------------------------------------------------------


@dataclass
class TerminationCfg:
    time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
    unstable: DoneTerm = term(
        DoneTerm,
        func=lambda env: torch.isnan(env.sim.data.qpos).any(dim=-1)
        | torch.isinf(env.sim.data.qpos).any(dim=-1),
    )
    cylinder_falls: DoneTerm = term(
        DoneTerm,
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "limit_angle": math.radians(75.0),
        },
    )


@dataclass
class EventCfg:
    reset_robot_joints: EventTerm = term(
        EventTerm,
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_robot_root: EventTerm = term(
        EventTerm,
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_object: EventTerm = term(
        EventTerm,
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


# -----------------------------------------------------------------------------
# Simulation configuration
# -----------------------------------------------------------------------------

SIM_CFG = SimulationCfg(
    nconmax=50000,
    njmax=1024,
    mujoco=MujocoCfg(timestep=0.008),
)  # pyright: ignore[reportGeneralTypeIssues]


# -----------------------------------------------------------------------------
# Curriculum logic
# -----------------------------------------------------------------------------

_STAGE_CONTROLLED_TERMS = (
    "reach",
    "gripper_open",
    "touch",
    "gripper_close",
    "grasp",
    "lift",
    "upreach",
)
_ALWAYS_ACTIVE_TERMS = ("wrist_roll_cost",)
_STAGE_METRIC_NAMES = ("reach", "touch", "grasp", "lift")

_STAGE_MULTIPLIERS: Tuple[Dict[str, float], ...] = (
    {"reach": 2.0, "gripper_open": 1.0},
    {"reach": 1.0, "touch": 2.0, "gripper_close": 1.0},
    {"touch": 1.0, "gripper_close": 1.0, "grasp": 5.0},  # Emphasize closing and grasping
    {"grasp": 2.0, "lift": 3.0, "upreach": 1.0},  # Emphasize lift when grasping
)
# % complete reach, touch, lift based on success thresholds
_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "reach": 0.7, 
    "touch": 0.8,
    "grasp": 0.3,  # Lower threshold - easier to progress
    "lift": 0.3,   # Lower threshold - easier to progress
}

_REACH_SUCCESS_THRESHOLD = 0.01 #m 10mm
_TOUCH_SUCCESS_THRESHOLD = 0.015 #m 20mm
_LIFT_SUCCESS_HEIGHT = 0.03 #m 30mm


def _find_checkpoint_directory() -> Path | None:
    """Try to find the checkpoint directory by looking for model checkpoint files.
    
    Returns the directory containing the most recently modified checkpoint file.
    If a curriculum_state.pt file exists, prioritizes directories with state files.
    This ensures we use the same directory as the actual checkpoint being used.
    """
    log_root = Path("logs") / "rsl_rl"
    if not log_root.exists():
        return None
    
    # First, check if there are any existing curriculum state files
    # If so, use the directory with the most recent state file
    latest_state_file = None
    latest_state_time = 0
    
    for exp_dir in log_root.iterdir():
        if not exp_dir.is_dir():
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            state_file = run_dir / "curriculum_state.pt"
            if state_file.exists():
                try:
                    mtime = state_file.stat().st_mtime
                    if mtime > latest_state_time:
                        latest_state_time = mtime
                        latest_state_file = state_file
                except OSError:
                    continue
    
    # If we found a state file, use its directory (this is likely the active run)
    if latest_state_file is not None:
        return latest_state_file.parent
    
    # Otherwise, look for the most recently modified checkpoint FILE
    latest_checkpoint = None
    latest_time = 0
    
    for exp_dir in log_root.iterdir():
        if not exp_dir.is_dir():
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            # Check if this directory contains checkpoint files
            checkpoints = list(run_dir.glob("model_*.pt"))
            for ckpt in checkpoints:
                try:
                    mtime = ckpt.stat().st_mtime
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_checkpoint = ckpt
                except OSError:
                    # Skip files we can't access
                    continue
    
    # Return the directory containing the most recent checkpoint
    if latest_checkpoint is not None:
        return latest_checkpoint.parent
    
    return None


def _ensure_curriculum_state(env: "ManagerBasedRlEnv") -> None:
    if not hasattr(env, "_curriculum_stage"):
        env._curriculum_stage = 0  # type: ignore[attr-defined]
        env._curriculum_metrics = {name: 0.0 for name in _STAGE_METRIC_NAMES}  # type: ignore[attr-defined]
    if not hasattr(env, "_curriculum_stage_initialized"):
        env._curriculum_stage_initialized = False  # type: ignore[attr-defined]


def _compute_success_metrics(env: "ManagerBasedRlEnv") -> Dict[str, float]:
    stage = int(getattr(env, "_curriculum_stage", 0))
    metrics: Dict[str, float] = {"reach": 0.0, "touch": 0.0, "grasp": 0.0, "lift": 0.0}

    if stage <= 0:
        reach_delta = _object_middle_pos(env) - _gripper_mid_pos(env)
        reach_success = (torch.norm(reach_delta, dim=-1) <= _REACH_SUCCESS_THRESHOLD).float()
        metrics["reach"] = reach_success.mean().item()
    elif stage == 1:
        object_surface1_pos = _object_surface1_pos(env)
        object_surface2_pos = _object_surface2_pos(env)
        touch_point1_pos = _touch_point1_pos(env)
        touch_point2_pos = _touch_point2_pos(env)
        
        delta1 = object_surface1_pos - touch_point1_pos
        delta2 = object_surface2_pos - touch_point2_pos
        
        distance1 = torch.norm(delta1, dim=-1)
        distance2 = torch.norm(delta2, dim=-1)
        
        # Use OR condition to match reward function - reward is given when either touch point is close
        touch_success = ((distance1 <= _GRIPPER_APPROACH_THRESHOLD) | (distance2 <= _GRIPPER_APPROACH_THRESHOLD)).float()
        metrics["touch"] = touch_success.mean().item()
    elif stage == 2:
        # Simplified: just need any force on either finger
        force_tp1, force_tp2 = _gripper_contact_forces(env)
        max_force = torch.maximum(force_tp1, force_tp2)
        grasp_success = max_force >= _GRASP_MIN_FORCE
        metrics["grasp"] = grasp_success.float().mean().item()
    else:
        obj = env.scene["object"]
        body_id = _object_body_index(env, "blue_cylinder")
        obj_height = obj.data.body_link_pos_w[:, body_id, 2]
        base_height = obj_height.new_tensor(obj.cfg.init_state.pos[2])
        lift_success = (obj_height - base_height >= _LIFT_SUCCESS_HEIGHT).float()
        metrics["lift"] = lift_success.mean().item()

    return metrics


def _cache_base_reward_weights(env: "ManagerBasedRlEnv") -> Dict[str, float]:
    if not hasattr(env, "_curriculum_base_reward_weights"):
        reward_manager = env.reward_manager
        weights: Dict[str, float] = {}
        for term in (*_STAGE_CONTROLLED_TERMS, *_ALWAYS_ACTIVE_TERMS):
            if term in reward_manager.active_terms:
                weights[term] = reward_manager.get_term_cfg(term).weight
        env._curriculum_base_reward_weights = weights  # type: ignore[attr-defined]
    return env._curriculum_base_reward_weights  # type: ignore[attr-defined]


def _apply_reward_stage(env: "ManagerBasedRlEnv", stage: int) -> None:
    stage = max(0, min(stage, len(_STAGE_MULTIPLIERS) - 1))
    base_weights = _cache_base_reward_weights(env)
    reward_manager = env.reward_manager
    multipliers = _STAGE_MULTIPLIERS[stage]

    for term, base_weight in base_weights.items():
        if term in _STAGE_CONTROLLED_TERMS:
            weight = base_weight * multipliers.get(term, 0.0)
        else:
            weight = base_weight
        reward_manager.get_term_cfg(term).weight = weight


def _curriculum_reward_schedule(
    env: "ManagerBasedRlEnv",
    env_ids,
    *,
    ema_alpha: float = 0.05,
    state_path: str | None = None,
    resume_curriculum: bool = True,
) -> torch.Tensor:
    del env_ids

    # Auto-discover checkpoint directory if state_path not provided
    # Keep trying until we find a checkpoint directory (useful when starting fresh training)
    if state_path is None:
        if not hasattr(env, "_curriculum_state_path"):
            checkpoint_dir = _find_checkpoint_directory()
            if checkpoint_dir is not None:
                state_path = str(checkpoint_dir / "curriculum_state.pt")
        else:
            # Use the already discovered path
            state_path = str(env._curriculum_state_path)  # type: ignore[attr-defined]
    
    # Load curriculum state if resuming and state file exists
    if state_path is not None:
        state_path_obj = Path(state_path)
        if not getattr(env, "_curriculum_state_loaded", False):
            if resume_curriculum and state_path_obj.exists():
                try:
                    data = torch.load(state_path_obj, map_location="cpu")
                    env._curriculum_stage = int(data.get("stage", 0))  # type: ignore[attr-defined]
                    env._curriculum_metrics = data.get("metrics", {name: 0.0 for name in _STAGE_METRIC_NAMES})  # type: ignore[attr-defined]
                    env._curriculum_stage_initialized = False  # type: ignore[attr-defined]  # Will be set to True after applying stage
                    _apply_reward_stage(env, env._curriculum_stage)  # type: ignore[attr-defined]
                    print(f"[INFO] Loaded curriculum state from {state_path_obj}: stage={env._curriculum_stage}")  # type: ignore[attr-defined]
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Failed to load curriculum state from {state_path_obj}: {exc}")
            elif not resume_curriculum:
                # Explicitly reset curriculum when not resuming
                if hasattr(env, "_curriculum_stage"):
                    env._curriculum_stage = 0  # type: ignore[attr-defined]
                if hasattr(env, "_curriculum_metrics"):
                    env._curriculum_metrics = {name: 0.0 for name in _STAGE_METRIC_NAMES}  # type: ignore[attr-defined]
                print(f"[INFO] Resetting curriculum (resume_curriculum=False)")
            env._curriculum_state_loaded = True  # type: ignore[attr-defined]
        
        # Set path for saving (even if we didn't load)
        if not hasattr(env, "_curriculum_state_path"):
            state_path_obj.parent.mkdir(parents=True, exist_ok=True)
            env._curriculum_state_path = state_path_obj  # type: ignore[attr-defined]

    _ensure_curriculum_state(env)

    metrics = _compute_success_metrics(env)

    for name in _STAGE_METRIC_NAMES:
        prev = env._curriculum_metrics.get(name, 0.0)  # type: ignore[attr-defined]
        env._curriculum_metrics[name] = (1.0 - ema_alpha) * prev + ema_alpha * metrics[name]  # type: ignore[attr-defined]

    stage = env._curriculum_stage  # type: ignore[attr-defined]
    metrics_ema = env._curriculum_metrics  # type: ignore[attr-defined]

    thresholds = _DEFAULT_THRESHOLDS

    if stage == 0 and metrics_ema["reach"] >= thresholds["reach"]:
        stage = 1
    elif stage == 1 and metrics_ema["touch"] >= thresholds["touch"]:
        stage = 2
    elif stage == 2 and metrics_ema["grasp"] >= thresholds["grasp"]:
        stage = 3
    elif stage >= 3 and metrics_ema["lift"] >= thresholds["lift"]:
        stage = 3
    stage = min(stage, len(_STAGE_MULTIPLIERS) - 1)

    if stage != env._curriculum_stage or not env._curriculum_stage_initialized:  # type: ignore[attr-defined]
        env._curriculum_stage = stage  # type: ignore[attr-defined]
        env._curriculum_stage_initialized = True  # type: ignore[attr-defined]
        _apply_reward_stage(env, stage)

    # Save curriculum state if we have a path
    if hasattr(env, "_curriculum_state_path"):
        state_path_obj = env._curriculum_state_path  # type: ignore[attr-defined]
        try:
            torch.save(
                {
                    "stage": int(stage),
                    "metrics": metrics_ema,
                },
                state_path_obj,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to save curriculum state to {state_path_obj}: {exc}")

    extras = getattr(env, "extras", None)
    if extras is None:
        env.extras = {"log": {}}
        extras = env.extras
    extras.setdefault("log", {})
    extras["log"].update(
        {
            "Curriculum/stage": float(env._curriculum_stage),  # type: ignore[attr-defined]
            "Curriculum/reach_success_ema": metrics_ema["reach"],
            "Curriculum/touch_success_ema": metrics_ema["touch"],
            "Curriculum/grasp_success_ema": metrics_ema["grasp"],
            "Curriculum/lift_success_ema": metrics_ema["lift"],
        }
    )

    return torch.tensor(float(stage), device=env.device)


# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------


@dataclass
class CurriculumCfg:
    reward_schedule: CurrTerm = term(
        CurrTerm,
        func=_curriculum_reward_schedule,
        params={
            "ema_alpha": 0.05,
        },
    )


@dataclass
class PPSimpleEnvCfg(ManagerBasedRlEnvCfg):
    scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
    observations: ObservationCfg = field(default_factory=ObservationCfg)
    actions: ActionCfg = field(default_factory=ActionCfg)
    rewards: RewardCfg = field(default_factory=RewardCfg)
    terminations: TerminationCfg = field(default_factory=TerminationCfg)
    events: EventCfg = field(default_factory=EventCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
    sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
    decimation: int = 2
    episode_length_s: float = 10.0


@dataclass
class PPSimpleEnvCfgPlay(PPSimpleEnvCfg):
    episode_length_s: float = 1e6

