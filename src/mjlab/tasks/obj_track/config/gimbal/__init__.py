"""Register gimbal object tracking task."""

from mjlab.tasks.registry import register_mjlab_task
from .env_cfgs import gimbal_obj_track_env_cfg
from .rl_cfg import gimbal_obj_track_rl_cfg

register_mjlab_task(
  task_id="Mjlab-ObjTrack-Gimbal",
  env_cfg=gimbal_obj_track_env_cfg(),
  play_env_cfg=gimbal_obj_track_env_cfg(),
  rl_cfg=gimbal_obj_track_rl_cfg(),
)
