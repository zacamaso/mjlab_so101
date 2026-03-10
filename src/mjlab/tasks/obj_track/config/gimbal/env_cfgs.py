"""Environment configurations for gimbal object tracking."""

from __future__ import annotations

from dataclasses import replace

from mjlab.asset_zoo.robots.gimbal.gimbal_constants import get_gimbal_robot_cfg
from mjlab.asset_zoo.objects.poi import get_poi_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.obj_track.obj_track_env_cfg import make_obj_track_env_cfg


def gimbal_obj_track_env_cfg() -> ManagerBasedRlEnvCfg:
  """Get gimbal object tracking environment configuration."""
  cfg = make_obj_track_env_cfg()

  # Set robot and POI.
  cfg.scene.entities["robot"] = get_gimbal_robot_cfg()
  cfg.scene.entities["poi"] = get_poi_cfg()

  return cfg
