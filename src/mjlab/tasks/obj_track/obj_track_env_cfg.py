"""Object track task configuration for the gimbal robot."""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
  ActionTermCfg,
  CommandTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.obj_track import mdp
from mjlab.tasks.obj_track.mdp import ObjTrackCommandCfg
from mjlab.sensor.camera_sensor import CameraSensorCfg
from mjlab.viewer import ViewerConfig


def make_obj_track_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base object tracking task configuration."""

  ##
  # Observations
  ##

  policy_terms = {
    "pixels": ObservationTermCfg(
      func=mdp.camera_pixels,
      params={"sensor_name": "robot/gimbal_cam", "data_type": "rgb"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
  }

  ##
  # Rewards
  ##

  reward_terms = {
    "pointing_reward": RewardTermCfg(
      func=mdp.pointing_error_reward,
      weight=10.0,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "target_cfg": SceneEntityCfg("poi"),
        "camera_name": "gimbal_cam",
      },
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
    ),
    "joint_vel_penalty": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=-0.001,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
  }

  ##
  # Terminations
  ##

  termination_terms = {
    "time_out": TerminationTermCfg(
      func=mdp.time_out,
      time_out=True,
    ),
  }

  ##
  # Events (Resets)
  ##

  event_terms = {
    "reset_robot": EventTermCfg(
      func=mdp.reset_scene_to_default,
      mode="reset",
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "reset_poi": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("poi"),
        "pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (0.5, 2.0)},
      },
    ),
  }

  ##
  # Scene
  ##

  scene = SceneCfg(
    num_envs=4096,
    env_spacing=4.0,
    entities={
      "robot": None,  # Set by robot-specific config.
      "poi": None,
    },
  )

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations={"policy": ObservationGroupCfg(terms=policy_terms)},
    actions={
      "arm": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        use_default_offset=True,
      )
    },
    rewards=reward_terms,
    terminations=termination_terms,
    events=event_terms,
    commands={
      "track_poi": ObjTrackCommandCfg(
        entity_name="poi",
        resampling_time_range=(10.0, 10.0),
      )
    },
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.005,
      ),
    ),
    viewer=ViewerConfig(lookat=(0.0, 0.0, 0.5), distance=3.0),
    decimation=1,
    episode_length_s=10.0,
  )
