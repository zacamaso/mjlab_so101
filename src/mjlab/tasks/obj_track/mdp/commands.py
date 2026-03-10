"""Object track commands."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class ObjTrackCommand(CommandTerm):
  """Command term for object tracking.

  This is a placeholder that can be used to track metadata or provide
  high-level goals. If the agent only sees pixels, this might not
  be directly fed to the policy.
  """

  def __init__(self, cfg: ObjTrackCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # We rely on the event term to move the POI in the simulation.
    # We can mirror its position here if needed for observations.
    poi = self._env.scene[self.cfg.entity_name]
    self.target_pos[env_ids] = poi.data.root_link_pos_w[env_ids]

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class ObjTrackCommandCfg(CommandTermCfg):
  entity_name: str
  resampling_time_range: tuple[float, float]

  def build(self, env: ManagerBasedRlEnv) -> ObjTrackCommand:
    return ObjTrackCommand(self, env)
