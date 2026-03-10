"""Object track rewards."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def pointing_error_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  target_cfg: SceneEntityCfg,
  camera_name: str,
) -> torch.Tensor:
  """Reward for pointing the camera at the target.

  Args:
    env: The environment.
    asset_cfg: The robot entity config.
    target_cfg: The target entity config.
    camera_name: The name of the camera on the robot.

  Returns:
    The pointing reward.
  """
  robot = env.scene[asset_cfg.name]
  target = env.scene[target_cfg.name]

  # Get target position in world frame.
  target_pos_w = target.data.root_link_pos_w

  # Get camera index.
  cam_id = env.sim.model.camera(f"{asset_cfg.name}/{camera_name}").id

  # Get camera position and orientation from MuJoCo data.
  # Note: env.sim.data is the mjwarp.Data which has cam_xpos and cam_xmat.
  cam_pos_w = env.sim.data.cam_xpos[:, cam_id]
  cam_mat_w = env.sim.data.cam_xmat[:, cam_id].reshape(-1, 3, 3)

  # Camera forward vector in MuJoCo is -Z (the 3rd column of xmat negated).
  cam_forward_w = -cam_mat_w[:, :, 2]

  # Vector from camera to target.
  vec_to_target = target_pos_w - cam_pos_w
  dist = torch.norm(vec_to_target, dim=-1, keepdim=True)
  vec_to_target_norm = vec_to_target / (dist + 1e-6)

  # Cosine similarity.
  cos_sim = torch.sum(cam_forward_w * vec_to_target_norm, dim=-1)

  # Reward is positive and increases as similarity approaches 1.
  return (cos_sim + 1.0) / 2.0
