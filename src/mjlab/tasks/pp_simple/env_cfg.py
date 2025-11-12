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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, cast

import torch  # pyright: ignore[reportMissingImports]

from mjlab.asset_zoo.misc_assets.blue_cylinder.blue_cylinder_constants import (
    BLUE_CYLINDER_CFG,
)
from mjlab.asset_zoo.robots.SO_101.so101_constants import SO101_ROBOT_CFG
from mjlab.envs import ManagerBasedEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.utils.spec_config import ContactSensorCfg
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


# =============================================================================
# HYPERPARAMETERS
# =============================================================================
# All tunable hyperparameters are organized here for easy modification.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Scene & Simulation Hyperparameters
# -----------------------------------------------------------------------------
NUM_ENVS = 1024  # Number of parallel environments
ENV_SPACING = 0.25  # Spacing between environments (meters)
SIM_TIMESTEP = 0.008  # Simulation timestep (seconds)
SIM_NCONMAX = 50000  # Maximum number of contacts
SIM_NJMAX = 1024  # Maximum number of constraints
EPISODE_LENGTH_S = 10.0  # Episode length in seconds
DECIMATION = 2  # Action decimation factor

# -----------------------------------------------------------------------------
# Object Geometry Hyperparameters
# -----------------------------------------------------------------------------
OBJECT_RADIUS = 0.0175  # Object radius (meters)
OBJECT_HALF_HEIGHT = 0.035 / 2.0  # Object half-height (meters)

# -----------------------------------------------------------------------------
# Gripper & Touch Hyperparameters
# -----------------------------------------------------------------------------
GRIPPER_APPROACH_THRESHOLD = 0.008  # Distance threshold for touch reward (meters)
TOUCH_THRESHOLD = 0.0005  # Distance threshold for touching detection (meters)
TOUCH_SUCCESS_THRESHOLD = 0.02  # Distance threshold for touch success metric (meters)
GRIPPER_OPEN_TARGET_GAP = 0.035  # Target gap when opening gripper (meters)
GRIPPER_CLOSE_TARGET_GAP = 0.0001  # Target gap when closing/touching (meters)
GRIPPER_CLOSE_GAP_REWARD_WEIGHT = 0.5  # Weight for gap reward when touching

# -----------------------------------------------------------------------------
# Grasp Hyperparameters
# -----------------------------------------------------------------------------
GRASP_MIN_FORCE = 0.005  # Minimum force threshold for grasp success (Newtons)
GRASP_FORCE_SATURATION = 0.2  # Force value at which grasp reward saturates (Newtons)
GRASP_BOTH_CONTACTING_THRESHOLD = 0.05  # Force threshold for "both contacting" bonus (Newtons)
GRASP_BALANCE_COEFFICIENT = 2.0  # Coefficient for balance bonus exponential
GRASP_BASE_REWARD_WEIGHT = 0.7*1000  # Base reward weight (single finger)
GRASP_BALANCED_REWARD_WEIGHT = 0.3*1000  # Additional reward weight when both fingers contact

# -----------------------------------------------------------------------------
# Reward Function Hyperparameters
# -----------------------------------------------------------------------------

# Reach reward
REACH_XY_WEIGHT = 50.0  # Exponential weight for XY distance
REACH_Z_WEIGHT = 15.0  # Exponential weight for Z distance
REACH_TP_WEIGHT = 10.0  # Exponential weight for touch point distance
REACH_SHARP_BONUS_WEIGHT = 20.0  # Exponential weight for sharp bonus
REACH_SHARP_BONUS_SCALE = 0.75  # Scale factor for sharp bonus
REACH_NEAR_MASK_THRESHOLD = 0.05  # Distance threshold for near mask (meters)
REACH_XY_VELOCITY_WEIGHT = 0.1  # Weight for XY velocity component
REACH_Z_VELOCITY_WEIGHT = 0.02  # Weight for Z velocity component
REACH_VELOCITY_MAX_BONUS = 0.25  # Maximum velocity bonus
REACH_REWARD_NORMALIZATION = 3.0  # Normalization factor for reach reward

# Touch reward
TOUCH_EXP_COEFFICIENT = 500.0  # Exponential coefficient for touch reward
TOUCH_BONUS_1 = 80.0  # Bonus value for touch point 1
TOUCH_BONUS_2 = 60.0  # Bonus value for touch point 2

# Open gripper reward
GRIPPER_OPEN_EXPLORATION_SCALE = 5.0  # Scale factor for exploration bonus
GRIPPER_OPEN_EXPLORATION_MAX = 0.5  # Maximum exploration bonus

# Close gripper reward
GRIPPER_CLOSE_GAP_EXP_COEFFICIENT = 50.0  # Exponential coefficient for gap reward

# Lift reward
LIFT_MAX_HEIGHT = 0.05  # Maximum height for lift reward (meters)
SETTLE_VELOCITY_THRESHOLD = 0.01  # Velocity threshold for considering object settled (m/s)
SETTLE_STEPS_REQUIRED = 10  # Number of consecutive steps below threshold to consider settled

# Upreach reward
UPREACH_MAX_HEIGHT_GAIN = 0.1  # Maximum height gain for upreach reward (meters)

# Wrist roll cost
WRIST_ROLL_EXP_COEFFICIENT = 200.0  # Exponential coefficient for wrist roll cost

# -----------------------------------------------------------------------------
# Reward Weight Hyperparameters (Base weights, multiplied by curriculum)
# -----------------------------------------------------------------------------
REWARD_WEIGHT_REACH = 5.0
REWARD_WEIGHT_TOUCH = 40.0
REWARD_WEIGHT_GRIPPER_OPEN = 1.0
REWARD_WEIGHT_GRIPPER_CLOSE = 50.0
REWARD_WEIGHT_GRASP = 1500.0
REWARD_WEIGHT_LIFT = 1.0
REWARD_WEIGHT_UPREACH = 1.0
REWARD_WEIGHT_WRIST_ROLL_COST = 120.0

# -----------------------------------------------------------------------------
# Curriculum Hyperparameters
# -----------------------------------------------------------------------------

# Stage multipliers: Each stage multiplies base reward weights
# Stage 0: Reach + Open gripper
# Stage 1: Touch + Close gripper (Stage 0 stays active)
# Stage 2: Grasp (prior stages stay active)
# Stage 3: Lift (prior stages stay active)
STAGE_MULTIPLIERS: Tuple[Dict[str, float], ...] = (
    {"reach": 2.0, "gripper_open": 1.0},  # Stage 0
    {"reach": 2.0, "touch": 2.5, "gripper_close": 1.0},  # Stage 1
    {"touch": 4.0, "gripper_close": 2.0, "grasp": 0.4},  # Stage 2: Emphasize closing to generate forces
    {"touch": 4.0, "grasp": 0.2, "lift": 2000.0, "upreach": 100.0},  # Stage 3
)

# Success thresholds: EMA success rate required to progress to next stage
CURRICULUM_THRESHOLD_REACH = 0.7  # 70% success rate
CURRICULUM_THRESHOLD_TOUCH = 0.6  # 60% success rate
CURRICULUM_THRESHOLD_GRASP = 0.3  # 30% success rate
CURRICULUM_THRESHOLD_LIFT = 0.3  # 30% success rate

# Success metric thresholds: Physical thresholds for computing success rates
REACH_SUCCESS_THRESHOLD = 0.01  # Distance threshold (meters, 10mm)
LIFT_SUCCESS_HEIGHT = 0.03  # Height threshold (meters, 30mm)

# Curriculum EMA
CURRICULUM_EMA_ALPHA = 0.05  # EMA smoothing factor for success metrics

# -----------------------------------------------------------------------------
# Action Hyperparameters
# -----------------------------------------------------------------------------
ACTION_SCALE = 0.5  # Scale factor for joint position actions
ACTION_USE_DEFAULT_OFFSET = True  # Whether to use default joint positions as offset

# -----------------------------------------------------------------------------
# Termination Hyperparameters
# -----------------------------------------------------------------------------
TERMINATION_CYLINDER_FALL_ANGLE_DEG = 75.0  # Maximum angle before termination (degrees)

# -----------------------------------------------------------------------------
# Event (Reset) Hyperparameters
# -----------------------------------------------------------------------------
# Object reset pose ranges
RESET_OBJECT_X_RANGE = (-0.02, 0.02)  # X position range (meters)
RESET_OBJECT_Y_RANGE = (-0.02, 0.02)  # Y position range (meters)
RESET_OBJECT_Z_RANGE = (0.0, 0.0)  # Z position range (meters)
RESET_OBJECT_ROLL_RANGE = (0.0, 0.0)  # Roll angle range (radians)
RESET_OBJECT_PITCH_RANGE = (0.0, 0.0)  # Pitch angle range (radians)
RESET_OBJECT_YAW_RANGE = (-0.1, 0.1)  # Yaw angle range (radians)

# Robot reset (all zeros for deterministic reset)
RESET_ROBOT_POSE_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}
RESET_ROBOT_VELOCITY_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}

# =============================================================================
# END OF HYPERPARAMETERS
# =============================================================================


# -----------------------------------------------------------------------------
# Scene configuration
# -----------------------------------------------------------------------------

# Add contact sensors to robot for efficient force detection
# Using body1 without geom2 to detect any contact with the touch point bodies
# This avoids cross-entity reference issues since sensors are added before scene merge
_GRIPPER_CONTACT_SENSORS = (
    ContactSensorCfg(
        name="touch_point1_contact",
        body1="touch_point1",
        num=1,
        data=("force",),  # Get force magnitude
        reduce="netforce",  # Net force magnitude
    ),
    ContactSensorCfg(
        name="touch_point2_contact",
        body1="touch_point2",
        num=1,
        data=("force",),  # Get force magnitude
        reduce="netforce",  # Net force magnitude
    ),
)

# Create robot config with contact sensors
_ROBOT_CFG_WITH_SENSORS = replace(SO101_ROBOT_CFG, sensors=_GRIPPER_CONTACT_SENSORS)

SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(terrain_type="plane", env_spacing=ENV_SPACING),
    num_envs=NUM_ENVS,
    entities={
        "robot": _ROBOT_CFG_WITH_SENSORS,
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


def _gripper_contact_forces(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contact force magnitudes on each fingertip using efficient contact sensors.
    
    Forces are clamped to prevent numerical instability and handle NaN/inf values.
    """
    robot = env.scene["robot"]
    device = robot.data.body_link_pos_w.device

    # Use contact sensors for efficient force detection
    try:
        tp1_force = robot.data.sensor_data["touch_point1_contact"][:, 0]  # Net force magnitude
        tp2_force = robot.data.sensor_data["touch_point2_contact"][:, 0]  # Net force magnitude
    except (KeyError, AttributeError):
        # Fallback to zeros if sensors not available (shouldn't happen with proper config)
        zeros = torch.zeros(env.num_envs, device=device)
        return zeros, zeros

    # Clamp forces to prevent numerical instability
    # Max force of 100N should be more than enough for grasping
    max_force = 100.0
    tp1_force = torch.clamp(tp1_force, min=0.0, max=max_force)
    tp2_force = torch.clamp(tp2_force, min=0.0, max=max_force)
    
    # Handle NaN/inf values (replace with zeros)
    tp1_force = torch.where(torch.isfinite(tp1_force), tp1_force, torch.zeros_like(tp1_force))
    tp2_force = torch.where(torch.isfinite(tp2_force), tp2_force, torch.zeros_like(tp2_force))

    return tp1_force, tp2_force


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
    offset = object_pos.new_tensor([0.0, -OBJECT_RADIUS, OBJECT_HALF_HEIGHT])
    return object_pos + offset


def _object_surface2_pos(env: ManagerBasedEnv) -> torch.Tensor:
    object_pos = _object_collision_pos(env)
    offset = object_pos.new_tensor([0.0, OBJECT_RADIUS, OBJECT_HALF_HEIGHT])
    return object_pos + offset


def _object_middle_pos(env: ManagerBasedEnv) -> torch.Tensor:
    object_pos = _object_collision_pos(env)
    offset = object_pos.new_tensor([0.0, 0.0, OBJECT_HALF_HEIGHT])
    return object_pos + offset


def _gripper_gap(env: ManagerBasedEnv) -> torch.Tensor:
    """Calculate distance between the two gripper touch points."""
    return torch.norm(_touch_point1_pos(env) - _touch_point2_pos(env), dim=-1)


def _touch_point_to_surface_distances(env: ManagerBasedEnv) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate distances from each touch point to its corresponding object surface.
    
    Returns:
        (dist1, dist2): Distances from touch_point1 to surface1 and touch_point2 to surface2
        Shape: ([num_envs], [num_envs]) for use in rewards/conditions
    """
    surface1 = _object_surface1_pos(env)
    surface2 = _object_surface2_pos(env)
    tp1 = _touch_point1_pos(env)
    tp2 = _touch_point2_pos(env)
    
    delta1 = surface1 - tp1
    delta2 = surface2 - tp2
    
    return torch.norm(delta1, dim=-1), torch.norm(delta2, dim=-1)


def _gripper_to_object_distance(env: ManagerBasedEnv) -> torch.Tensor:
    """Calculate distance from gripper center to object center."""
    gripper_pos = _gripper_mid_pos(env)
    object_pos = _object_middle_pos(env)
    delta = object_pos - gripper_pos
    return torch.norm(delta, dim=-1)


def _is_touching(env: ManagerBasedEnv, threshold: Optional[float] = None) -> torch.Tensor:
    """Check if both touch points are within threshold distance of object surfaces.
    
    Args:
        env: Environment instance
        threshold: Distance threshold (defaults to TOUCH_THRESHOLD)
    
    Returns:
        Boolean tensor indicating if touching (both points within threshold)
    """
    if threshold is None:
        threshold = TOUCH_THRESHOLD
    dist1, dist2 = _touch_point_to_surface_distances(env)
    return (dist1 <= threshold) & (dist2 <= threshold)


def _is_near_object(env: ManagerBasedEnv, threshold: Optional[float] = None) -> torch.Tensor:
    """Check if gripper is near object (for reach reward).
    
    Args:
        env: Environment instance
        threshold: Distance threshold (defaults to GRIPPER_APPROACH_THRESHOLD)
    
    Returns:
        Boolean tensor indicating if near object
    """
    if threshold is None:
        threshold = GRIPPER_APPROACH_THRESHOLD
    dist1, dist2 = _touch_point_to_surface_distances(env)
    # Near if either touch point is close (more forgiving)
    return (dist1 <= threshold) | (dist2 <= threshold)


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------


def _obs_object_to_gripper(env: ManagerBasedEnv) -> torch.Tensor:
    return _object_middle_pos(env) - _gripper_mid_pos(env)


def _obs_touch_dist1(env: ManagerBasedEnv) -> torch.Tensor:
    """Distance from touch point 1 to object surface 1. Shape: [num_envs, 1]"""
    dist1, _ = _touch_point_to_surface_distances(env)
    return dist1.unsqueeze(-1)


def _obs_touch_dist2(env: ManagerBasedEnv) -> torch.Tensor:
    """Distance from touch point 2 to object surface 2. Shape: [num_envs, 1]"""
    _, dist2 = _touch_point_to_surface_distances(env)
    return dist2.unsqueeze(-1)


def _obs_force_tp1(env: ManagerBasedEnv) -> torch.Tensor:
    """Contact force magnitude on touch point 1. Shape: [num_envs, 1]
    
    Forces are normalized and clamped to prevent extreme values from destabilizing policy.
    """
    force_tp1, _ = _gripper_contact_forces(env)
    # Normalize forces (same as reward) and clamp to reasonable range
    normalized_force = torch.clamp(force_tp1 / GRASP_FORCE_SATURATION, 0.0, 10.0)
    return normalized_force.unsqueeze(-1)


def _obs_force_tp2(env: ManagerBasedEnv) -> torch.Tensor:
    """Contact force magnitude on touch point 2. Shape: [num_envs, 1]
    
    Forces are normalized and clamped to prevent extreme values from destabilizing policy.
    """
    _, force_tp2 = _gripper_contact_forces(env)
    # Normalize forces (same as reward) and clamp to reasonable range
    normalized_force = torch.clamp(force_tp2 / GRASP_FORCE_SATURATION, 0.0, 10.0)
    return normalized_force.unsqueeze(-1)


@dataclass
class ObservationCfg:
    """Observation terms exposed to both policy and critic."""

    @dataclass
    class PolicyCfg(ObsGroup):
        joint_pos: ObsTerm = term(ObsTerm, func=mdp.joint_pos_rel)
        object_to_gripper: ObsTerm = term(ObsTerm, func=_obs_object_to_gripper)
        touch_dist1: ObsTerm = term(ObsTerm, func=_obs_touch_dist1)
        touch_dist2: ObsTerm = term(ObsTerm, func=_obs_touch_dist2)
        force_tp1: ObsTerm = term(ObsTerm, func=_obs_force_tp1)
        force_tp2: ObsTerm = term(ObsTerm, func=_obs_force_tp2)

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
        scale=ACTION_SCALE,
        use_default_offset=ACTION_USE_DEFAULT_OFFSET,
    )


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------


def _reward_reach(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward for reaching the object with gripper."""
    gripper_pos = _gripper_mid_pos(env)
    object_pos = _object_middle_pos(env)
    delta = object_pos - gripper_pos

    # Distance components
    xy_distance = torch.norm(delta[:, :2], dim=-1)
    z_distance = torch.abs(delta[:, 2])
    total_distance = torch.norm(delta, dim=-1)
    
    # Touch point distances (for fine-grained control)
    dist1, dist2 = _touch_point_to_surface_distances(env)
    avg_tp_distance = 0.5 * (dist1 + dist2)

    # Distance-based rewards
    xy_reward = torch.exp(-REACH_XY_WEIGHT * xy_distance)
    z_reward = torch.exp(-REACH_Z_WEIGHT * z_distance)
    tp_reward = torch.exp(-REACH_TP_WEIGHT * avg_tp_distance)
    sharp_bonus = torch.exp(-REACH_SHARP_BONUS_WEIGHT * total_distance)
    distance_reward = xy_reward * z_reward + REACH_SHARP_BONUS_SCALE * sharp_bonus + tp_reward

    # Velocity bonus (encourage moving toward object)
    geom_id = _robot_geom_index(env, "gripper_mid_point")
    robot = env.scene["robot"]
    gripper_vel = robot.data.geom_lin_vel_w[:, geom_id, :]

    xy_unit = delta[:, :2] / (xy_distance.unsqueeze(-1) + 1e-6)
    xy_vel_toward = torch.sum(gripper_vel[:, :2] * xy_unit, dim=-1)
    z_vel_toward = gripper_vel[:, 2] * torch.sign(delta[:, 2])

    near_mask = (total_distance < REACH_NEAR_MASK_THRESHOLD).float()
    velocity_components = (
        REACH_XY_VELOCITY_WEIGHT * torch.clamp(xy_vel_toward, min=0.0)
        + REACH_Z_VELOCITY_WEIGHT * torch.clamp(z_vel_toward, min=0.0)
    )
    velocity_bonus = near_mask * torch.clamp(velocity_components, max=REACH_VELOCITY_MAX_BONUS)

    raw_reward = distance_reward + velocity_bonus
    return torch.clamp(raw_reward / REACH_REWARD_NORMALIZATION, 0.0, 1.0)


def _reward_touch(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward for touching the object with gripper fingers."""
    dist1, dist2 = _touch_point_to_surface_distances(env)
    threshold = GRIPPER_APPROACH_THRESHOLD

    # Normalize distances to [0, 1] within threshold
    normalized1 = torch.clamp(dist1 / threshold, min=0.0, max=1.0)
    normalized2 = torch.clamp(dist2 / threshold, min=0.0, max=1.0)

    # Exponential reward when within threshold, zero otherwise
    touch_reward1 = torch.where(
        dist1 <= threshold,
        torch.exp(-TOUCH_EXP_COEFFICIENT * normalized1) + TOUCH_BONUS_1,
        torch.zeros_like(dist1),
    )
    touch_reward2 = torch.where(
        dist2 <= threshold,
        torch.exp(-TOUCH_EXP_COEFFICIENT * normalized2) + TOUCH_BONUS_2,
        torch.zeros_like(dist2),
    )

    touch_reward = touch_reward1 + touch_reward2
    return torch.clamp(touch_reward, 0.0, 1.0)


def _reward_open_gripper(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward for opening gripper (before touching object)."""
    rl_env = cast("ManagerBasedRlEnv", env)
    gap = _gripper_gap(env)

    max_reward = 1.0
    base_reward = torch.clamp((gap - GRIPPER_OPEN_TARGET_GAP) / GRIPPER_OPEN_TARGET_GAP, min=0.0, max=max_reward)

    # Track gap changes for exploration bonus
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

    exploration_bonus = torch.clamp(gap_delta * GRIPPER_OPEN_EXPLORATION_SCALE, max=GRIPPER_OPEN_EXPLORATION_MAX)

    # If touching, give max reward (gripper should be open before touching)
    touching = _is_touching(env, TOUCH_THRESHOLD)

    raw_reward = torch.where(
        touching,
        torch.full_like(base_reward, max_reward),
        base_reward + exploration_bonus,
    )

    return torch.clamp(raw_reward / max_reward, 0.0, 1.0)


def _reward_close_gripper(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward for closing gripper when near/touching object."""
    rl_env = cast("ManagerBasedRlEnv", env)
    dist1, dist2 = _touch_point_to_surface_distances(env)
    touching = _is_touching(env, TOUCH_THRESHOLD)
    near = _is_near_object(env, GRIPPER_APPROACH_THRESHOLD)  # More forgiving threshold

    # Track distance changes for progress reward
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

    progress_reward = (dist1_delta + dist2_delta) / (2.0 * TOUCH_THRESHOLD)
    progress_reward = torch.clamp(progress_reward, 0.0, 1.0)

    # Reward gripper closing when near object (not just touching)
    # This encourages closing while positioning, not just after touching
    gap = _gripper_gap(env)
    gap_reward = torch.exp(-GRIPPER_CLOSE_GAP_EXP_COEFFICIENT * torch.clamp(gap - GRIPPER_CLOSE_TARGET_GAP, min=0.0))
    
    # When near object, add gap reward (stronger when touching)
    gap_bonus = GRIPPER_CLOSE_GAP_REWARD_WEIGHT * gap_reward
    # Double the bonus when actually touching
    gap_bonus = torch.where(touching, gap_bonus * 5.0, gap_bonus)
    
    # Only apply gap bonus when near object
    final_reward = torch.where(
        near,
        torch.clamp(progress_reward + gap_bonus, 0.0, 1.0),
        progress_reward
    )
    
    return final_reward


def _reward_grasp(env: ManagerBasedEnv) -> torch.Tensor:
    """Simplified grasp reward that rewards any force presence."""
    force_tp1, force_tp2 = _gripper_contact_forces(env)
    
    # Reward based on average force (encourages both fingers to contact)
    avg_force = 0.5 * (force_tp1 + force_tp2)
    
    # Scale reward by force magnitude
    force_reward = torch.clamp(avg_force / GRASP_FORCE_SATURATION, 0.0, 1.0)
    
    # Bonus for balanced forces (both fingers contacting)
    both_contacting = (force_tp1 > GRASP_BOTH_CONTACTING_THRESHOLD) & (force_tp2 > GRASP_BOTH_CONTACTING_THRESHOLD)
    balance_bonus = torch.exp(-GRASP_BALANCE_COEFFICIENT * torch.abs(force_tp1 - force_tp2) / (avg_force + 0.01))
    
    # Multiply by touch detection to prevent exploiting self-closing
    # Use gradual touch factor based on distance to object surfaces
    dist1, dist2 = _touch_point_to_surface_distances(env)
    touch_factor1 = torch.exp(-dist1 / TOUCH_THRESHOLD)
    touch_factor2 = torch.exp(-dist2 / TOUCH_THRESHOLD)
    touch_factor = 0.5 * (touch_factor1 + touch_factor2)
    
    reward = force_reward * touch_factor * (GRASP_BASE_REWARD_WEIGHT + GRASP_BALANCED_REWARD_WEIGHT * both_contacting.float() * balance_bonus)
    return torch.clamp(reward, 0.0, 1.0)


def _reward_lift(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward for lifting object upward, only when grasping.
    
    Tracks the settled height of the object after it drops, then measures
    lift relative to that settled position.
    """
    rl_env = cast("ManagerBasedRlEnv", env)
    obj = env.scene["object"]
    body_id = _object_body_index(env, "blue_cylinder")
    obj_height = obj.data.body_link_pos_w[:, body_id, 2]
    obj_lin_vel = obj.data.root_link_vel_w[:, :3]  # Linear velocity (first 3 components)
    obj_vel_magnitude = torch.norm(obj_lin_vel, dim=-1)
    
    # Initialize tracking variables (store on env, not rl_env, for consistency)
    if not hasattr(env, "_object_settled_height"):
        env._object_settled_height = obj_height.clone()  # type: ignore[attr-defined]
        env._settle_step_counter = torch.zeros(env.num_envs, device=obj_height.device, dtype=torch.int32)  # type: ignore[attr-defined]
        env._object_settled = torch.zeros(env.num_envs, device=obj_height.device, dtype=torch.bool)  # type: ignore[attr-defined]
    else:
        # Ensure shapes match (in case num_envs changed)
        if env._object_settled_height.shape[0] != env.num_envs:  # type: ignore[attr-defined]
            env._object_settled_height = obj_height.clone()  # type: ignore[attr-defined]
            env._settle_step_counter = torch.zeros(env.num_envs, device=obj_height.device, dtype=torch.int32)  # type: ignore[attr-defined]
            env._object_settled = torch.zeros(env.num_envs, device=obj_height.device, dtype=torch.bool)  # type: ignore[attr-defined]
    
    # Reset on environment resets
    if hasattr(rl_env, "reset_buf"):
        reset_buf = getattr(rl_env, "reset_buf")
        if isinstance(reset_buf, torch.Tensor) and reset_buf.any():
            reset_mask = reset_buf.bool()
            env._object_settled_height = env._object_settled_height.clone()  # type: ignore[attr-defined]
            env._object_settled_height[reset_mask] = obj_height[reset_mask]  # type: ignore[attr-defined]
            env._settle_step_counter = env._settle_step_counter.clone()  # type: ignore[attr-defined]
            env._settle_step_counter[reset_mask] = 0  # type: ignore[attr-defined]
            env._object_settled = env._object_settled.clone()  # type: ignore[attr-defined]
            env._object_settled[reset_mask] = False  # type: ignore[attr-defined]
    
    # Check if object is settling (low velocity)
    is_low_velocity = obj_vel_magnitude < SETTLE_VELOCITY_THRESHOLD
    
    # Increment counter for low velocity, reset if velocity increases
    env._settle_step_counter = torch.where(  # type: ignore[attr-defined]
        is_low_velocity,
        env._settle_step_counter + 1,  # type: ignore[attr-defined]
        torch.zeros_like(env._settle_step_counter)  # type: ignore[attr-defined]
    )
    
    # Mark as settled if we've had low velocity for enough steps
    newly_settled = (env._settle_step_counter >= SETTLE_STEPS_REQUIRED) & (~env._object_settled)  # type: ignore[attr-defined]
    env._object_settled = env._object_settled | newly_settled  # type: ignore[attr-defined]
    
    # Update settled height when object first settles
    env._object_settled_height = torch.where(  # type: ignore[attr-defined]
        newly_settled.unsqueeze(-1),
        obj_height.unsqueeze(-1),
        env._object_settled_height.unsqueeze(-1)  # type: ignore[attr-defined]
    ).squeeze(-1)
    
    # Use settled height as baseline (or initial spawn height if not settled yet)
    base_height = torch.where(
        env._object_settled,  # type: ignore[attr-defined]
        env._object_settled_height,  # type: ignore[attr-defined]
        obj_height.new_tensor(obj.cfg.init_state.pos[2])
    )
    
    height_diff = obj_height - base_height
    
    # Only reward when grasping/touching the object
    touching = _is_touching(env, TOUCH_THRESHOLD)
    force_tp1, force_tp2 = _gripper_contact_forces(env)
    avg_force = 0.5 * (force_tp1 + force_tp2)
    grasping = (avg_force > GRASP_MIN_FORCE) | touching
    
    # Clamp height difference
    lifted = torch.clamp(height_diff, min=0.0, max=LIFT_MAX_HEIGHT)
    
    # Use exponential reward that's more sensitive to small movements
    # This gives meaningful reward even for tiny lifts (encourages initiation)
    exp_component = 1.0 - torch.exp(-20.0 * lifted / LIFT_MAX_HEIGHT)
    
    # Linear component for larger lifts
    linear_component = lifted / LIFT_MAX_HEIGHT
    
    # Combine: exponential dominates for small lifts, linear for larger
    lift_reward = 0.6 * exp_component + 0.4 * linear_component
    
    # Bonus for any upward movement when grasping (encourages initiation)
    any_lift_bonus = torch.clamp(lifted * 20.0, 0.0, 0.3)  # Max 0.3 bonus for tiny lifts
    
    lift_reward = lift_reward + any_lift_bonus
    
    # Only give reward when grasping
    lift_reward = lift_reward * grasping.float()
    
    return torch.clamp(lift_reward, 0.0, 1.0)


def _reward_upreach(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward for lifting gripper upward while touching/grasping object."""
    rl_env = cast("ManagerBasedRlEnv", env)
    
    # Check both touching and grasping
    touching = _is_touching(env, TOUCH_THRESHOLD)
    force_tp1, force_tp2 = _gripper_contact_forces(env)
    avg_force = 0.5 * (force_tp1 + force_tp2)
    grasping = (avg_force > GRASP_MIN_FORCE) | touching
    grasp_mask = grasping.float()

    robot = env.scene["robot"]
    geom_ids, _ = robot.find_geoms("gripper_mid_point", preserve_order=True)
    gripper_pos = robot.data.geom_pos_w[:, geom_ids[0], :]
    gripper_height = gripper_pos[:, 2]

    # Initialize or reset initial height
    if (
        not hasattr(rl_env, "_initial_gripper_height")
        or rl_env._initial_gripper_height.shape[0] != rl_env.num_envs
    ):
        rl_env._initial_gripper_height = gripper_height.clone()
    else:
        # Reset initial height on environment resets
        if hasattr(rl_env, "reset_buf"):
            reset_buf = getattr(rl_env, "reset_buf")
            if isinstance(reset_buf, torch.Tensor) and reset_buf.any():
                rl_env._initial_gripper_height = rl_env._initial_gripper_height.clone()
                rl_env._initial_gripper_height[reset_buf.bool()] = gripper_height[reset_buf.bool()]

    height_gain = torch.clamp(
        gripper_height - rl_env._initial_gripper_height, min=0.0, max=UPREACH_MAX_HEIGHT_GAIN
    )
    
    # Use exponential reward for small movements to encourage initiation
    exp_component = 1.0 - torch.exp(-20.0 * height_gain / UPREACH_MAX_HEIGHT_GAIN)
    # Linear component for larger movements
    linear_component = height_gain / UPREACH_MAX_HEIGHT_GAIN
    
    # Combine: exponential dominates for small movements
    lift_reward = 0.6 * exp_component + 0.4 * linear_component
    
    # Bonus for any upward movement when grasping (encourages initiation)
    any_lift_bonus = torch.clamp(height_gain * 20.0 / UPREACH_MAX_HEIGHT_GAIN, 0.0, 0.3)
    
    lift_reward = lift_reward + any_lift_bonus
    
    # Only give reward when grasping/touching
    up_reward = grasp_mask * lift_reward
    
    return torch.clamp(up_reward, 0.0, 1.0)


def _cost_wrist_roll(env: ManagerBasedEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos[:, 4]
    target = torch.as_tensor(0, device=joint_pos.device, dtype=joint_pos.dtype)
    deviation = torch.abs(joint_pos - target)
    reward = torch.exp(-WRIST_ROLL_EXP_COEFFICIENT * deviation)
    return -torch.clamp(reward, 0.0, 1.0)


@dataclass
class RewardCfg:
    reach: RewardTerm = term(RewardTerm, func=_reward_reach, weight=REWARD_WEIGHT_REACH)
    touch: RewardTerm = term(RewardTerm, func=_reward_touch, weight=REWARD_WEIGHT_TOUCH)
    gripper_open: RewardTerm = term(RewardTerm, func=_reward_open_gripper, weight=REWARD_WEIGHT_GRIPPER_OPEN)
    gripper_close: RewardTerm = term(RewardTerm, func=_reward_close_gripper, weight=REWARD_WEIGHT_GRIPPER_CLOSE)
    grasp: RewardTerm = term(RewardTerm, func=_reward_grasp, weight=REWARD_WEIGHT_GRASP)
    lift: RewardTerm = term(RewardTerm, func=_reward_lift, weight=REWARD_WEIGHT_LIFT)
    wrist_roll_cost: RewardTerm = term(RewardTerm, func=_cost_wrist_roll, weight=REWARD_WEIGHT_WRIST_ROLL_COST)
    upreach: RewardTerm = term(RewardTerm, func=_reward_upreach, weight=REWARD_WEIGHT_UPREACH)


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
            "limit_angle": math.radians(TERMINATION_CYLINDER_FALL_ANGLE_DEG),
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
            "pose_range": RESET_ROBOT_POSE_RANGE,
            "velocity_range": RESET_ROBOT_VELOCITY_RANGE,
        },
    )
    reset_object: EventTerm = term(
        EventTerm,
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {
                "x": RESET_OBJECT_X_RANGE,
                "y": RESET_OBJECT_Y_RANGE,
                "z": RESET_OBJECT_Z_RANGE,
                "roll": RESET_OBJECT_ROLL_RANGE,
                "pitch": RESET_OBJECT_PITCH_RANGE,
                "yaw": RESET_OBJECT_YAW_RANGE,
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
    nconmax=SIM_NCONMAX,
    njmax=SIM_NJMAX,
    mujoco=MujocoCfg(timestep=SIM_TIMESTEP),
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

_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "reach": CURRICULUM_THRESHOLD_REACH,
    "touch": CURRICULUM_THRESHOLD_TOUCH,
    "grasp": CURRICULUM_THRESHOLD_GRASP,
    "lift": CURRICULUM_THRESHOLD_LIFT,
}


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
        reach_success = (torch.norm(reach_delta, dim=-1) <= REACH_SUCCESS_THRESHOLD).float()
        metrics["reach"] = reach_success.mean().item()
    elif stage == 1:
        # Touch success: either touch point is close to its surface
        touch_success = _is_near_object(env, GRIPPER_APPROACH_THRESHOLD).float()
        metrics["touch"] = touch_success.mean().item()
    elif stage == 2:
        # Simplified: just need any force on either finger
        force_tp1, force_tp2 = _gripper_contact_forces(env)
        max_force = torch.maximum(force_tp1, force_tp2)
        grasp_success = max_force >= GRASP_MIN_FORCE
        metrics["grasp"] = grasp_success.float().mean().item()
    else:
        obj = env.scene["object"]
        body_id = _object_body_index(env, "blue_cylinder")
        obj_height = obj.data.body_link_pos_w[:, body_id, 2]
        
        # Use settled height if available (from lift reward tracking), otherwise use initial spawn height
        if hasattr(env, "_object_settled_height") and hasattr(env, "_object_settled"):
            base_height = torch.where(
                env._object_settled,
                env._object_settled_height,
                obj_height.new_tensor(obj.cfg.init_state.pos[2])
            )
        else:
            base_height = obj_height.new_tensor(obj.cfg.init_state.pos[2])
        
        lift_success = (obj_height - base_height >= LIFT_SUCCESS_HEIGHT).float()
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
    stage = max(0, min(stage, len(STAGE_MULTIPLIERS) - 1))
    base_weights = _cache_base_reward_weights(env)
    reward_manager = env.reward_manager
    multipliers = STAGE_MULTIPLIERS[stage]

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
    stage = min(stage, len(STAGE_MULTIPLIERS) - 1)

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
            "ema_alpha": CURRICULUM_EMA_ALPHA,
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
    decimation: int = DECIMATION
    episode_length_s: float = EPISODE_LENGTH_S


@dataclass
class PPSimpleEnvCfgPlay(PPSimpleEnvCfg):
    episode_length_s: float = 1e6

