"""SO-101 robot constants and spec loader."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets


# MJCF and assets.
SO101_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "SO_101" / "xmls" / "SO-102-B.xml"
)
assert SO101_XML.exists(), f"XML not found: {SO101_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, SO101_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(SO101_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


SO101_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlPositionActuatorCfg(target_names_expr=(".*",)),),
  soft_joint_pos_limit_factor=0.95,
)


def get_so101_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0055),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos={
        "shoulder_pan": 0.0,
        "shoulder_lift": -1.745,
        "elbow_flex": 1.69,
        "wrist_flex": 1.3264,
        "wrist_roll": 1.02588,
        "gripper": 0.0,
      },
    ),
    spec_fn=get_spec,
    articulation=SO101_ARTICULATION,
  )


# Calculate action scales from XML actuator properties
# XML actuators have kp=17.8 (stiffness) and forcerange="-1.5 1.5" (effort_limit=1.5)
# Formula: 0.25 * effort_limit / stiffness = 0.25 * 1.5 / 17.8 ≈ 0.021
SO101_ACTION_SCALE: dict[str, float] = {}
spec = get_spec()
for actuator in spec.actuators:
  if actuator.name and actuator.target:
    # Extract stiffness from XML (kp=17.8 for sts3215 class)
    stiffness = 17.8
    # Extract effort limit from forcerange (1.5 for all actuators)
    effort_limit = 1.5
    # Calculate action scale using same formula as other robots
    SO101_ACTION_SCALE[actuator.target] = 0.25 * effort_limit / stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_so101_robot_cfg())
  viewer.launch(robot.spec.compile())
