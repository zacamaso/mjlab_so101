"""POI (Point of Interest) entity definition."""

from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import MaterialCfg


def get_poi_cfg() -> EntityCfg:
  """Get POI (Point of Interest) configuration."""
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(1.0, 0.0, 0.5),
    ),
    spec_fn=poi_spec_fn,
  )


def poi_spec_fn(spec):
  """Add a mocap sphere for the POI."""
  body = spec.worldbody.add_body(name="poi", mocap=True)
  body.add_geom(
    name="poi_geom",
    type="sphere",
    size=[0.05],
    rgba=[1, 1, 0, 1],
  )
