from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import so101_lift_cube_env_cfg
from .rl_cfg import so101_lift_cube_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-SO101",
  env_cfg=so101_lift_cube_env_cfg(),
  play_env_cfg=so101_lift_cube_env_cfg(play=True),
  rl_cfg=so101_lift_cube_ppo_runner_cfg(),
)

