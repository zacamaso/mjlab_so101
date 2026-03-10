"""Object track observations."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def camera_pixels(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  data_type: Literal["rgb", "depth"] = "rgb",
) -> torch.Tensor:
  """Get camera sensor pixel data.

  Args:
    env: The environment.
    sensor_name: The name of the camera sensor.
    data_type: The type of camera data to return ("rgb" or "depth").

  Returns:
    The pixel data tensor.
  """
  sensor = env.sim.sensors[sensor_name]
  data = sensor.data()
  if data_type == "rgb":
    return data.rgb.float() / 255.0  # Normalize to [0, 1]
  elif data_type == "depth":
    return data.depth
  else:
    raise ValueError(f"Invalid camera data type: {data_type}")
