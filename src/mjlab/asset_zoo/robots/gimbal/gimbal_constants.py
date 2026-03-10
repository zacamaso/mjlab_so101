"""Gimbal robot definition as EntityCfg."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils import spec_config
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CameraCfg, CollisionCfg

_HERE = Path(__file__).parent

GIMBAL_XML: Path = _HERE / "xmls" / "gimbal.xml"
assert GIMBAL_XML.exists()


def get_assets(meshdir: str | None) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  assets_dir = _HERE / "assets"
  if assets_dir.exists():
    update_assets(assets, assets_dir, meshdir or "")
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GIMBAL_XML))
  spec.assets = get_assets(getattr(spec, "meshdir", None))
  return spec


EFFORT_LIMIT = 5.0
ARMATURE = 0.001
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 1.0
STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

GIMBAL_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("yaw", "pitch"),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE,
)

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.5),
  joint_pos={"yaw": 0.0, "pitch": 0.0},
  joint_vel={".*": 0.0},
)

GIMBAL_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GIMBAL_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)


def get_gimbal_robot_cfg() -> EntityCfg:
  """Get gimbal robot configuration."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(CollisionCfg(geom_names_expr=(".*",), condim=3, priority=1),),
    cameras=(
      CameraCfg(
        name="gimbal_cam",
        body="camera_body",
        pos=(0, 0, 0),
        quat=(0.707, 0, 0.707, 0),  # Look forward (+Y or +Z depending on orientation)
      ),
    ),
    spec_fn=get_spec,
    articulation=GIMBAL_ARTICULATION,
  )


GIMBAL_ACTION_SCALE: dict[str, float] = {}
for _a in GIMBAL_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  assert _e is not None
  for _n in _a.target_names_expr:
    GIMBAL_ACTION_SCALE[_n] = 0.25 * _e / _s
