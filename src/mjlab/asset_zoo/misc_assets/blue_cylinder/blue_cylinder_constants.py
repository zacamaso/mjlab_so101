"""Blue cylinder entity constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets


# Path to the blue cylinder STL file   /home/zaca/Documents/CODE/mjlab/src/mjlab/asset_zoo/misc_assets/blue_cylinder/xmls/assets/blue_cyclinder.stl
BLUE_CYLINDER_XML = (
    MJLAB_SRC_PATH / "asset_zoo" / "misc_assets" / "blue_cylinder" / "xmls" / "blue_cylinder.xml"
)
assert BLUE_CYLINDER_XML.exists(), f"STL file not found: {BLUE_CYLINDER_XML}"

def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, BLUE_CYLINDER_XML.parent / "assets", meshdir)
    return assets

def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(BLUE_CYLINDER_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


BLUE_CYLINDER_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(),  # No actuators for passive object
    soft_joint_pos_limit_factor=1.0,
)

BLUE_CYLINDER_CFG = EntityCfg(
    spec_fn=get_spec,
    articulation=BLUE_CYLINDER_ARTICULATION,
  init_state=EntityCfg.InitialStateCfg(
    pos=(0.3, 0.0, 0.08),  # Position in front of camera (above ground)
    rot=(1.0, 0.0, 0.0, 0.0),  # No rotation
    lin_vel=(0.0, 0.0, 0.0),  # No linear velocity
    ang_vel=(0.0, 0.0, 0.0),  # No angular velocity
  ),
)
