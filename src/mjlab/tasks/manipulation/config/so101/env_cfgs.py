import mujoco

from mjlab.asset_zoo.robots import (
  SO101_ACTION_SCALE,
  get_so101_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg


def get_cube_spec(cube_size: float = 0.01, mass: float = 0.00625) -> mujoco.MjSpec:
  # Cube is 50% smaller than YAM (0.02 -> 0.01)
  # Mass scales with volume: 0.05 * (0.5^3) = 0.00625
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=(0.8, 0.2, 0.2, 1.0),
  )
  return spec


def so101_lift_cube_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_so101_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = SO101_ACTION_SCALE

  assert cfg.commands is not None
  lift_command = cfg.commands["lift_height"]
  assert isinstance(lift_command, LiftingCommandCfg)
  assert lift_command.object_pose_range is not None
  # Bring target position 50% closer to arm (multiply ranges by 0.5)
  lift_command.object_pose_range.x = (0.2, 0.3)  # was (0.2, 0.4)
  lift_command.object_pose_range.y = (-0.12, 0.12)  # was (-0.2, 0.2)
  lift_command.object_pose_range.z = (0.015, 0.035)  # was (0.02, 0.05), also 50% smaller
  # Bring target sphere 50% closer and make it 50% smaller
  lift_command.target_position_range.x = (0.15, 0.25)  # was (0.3, 0.5)
  lift_command.target_position_range.y = (-0.1, 0.1)  # was (-0.2, 0.2)
  lift_command.target_position_range.z = (0.1, 0.2)  # was (0.2, 0.4)
  lift_command.viz.target_sphere_radius = 0.015  # was 0.03, 50% smaller

  # # Disable mid-episode resampling to synchronize cube reset with robot reset
  # lift_command.resampling_time_range = (1e9, 1e9)

  cfg.observations["policy"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "gripperframe",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("gripperframe",)

  fingertip_geoms = r"touch_point.*_geom_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensor pattern.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "moving_jaw_so101_v1"

  cfg.viewer.body_name = "base"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg

