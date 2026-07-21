from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def payload_dr_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    mass_event_name: str = "add_base_mass",
    com_event_name: str = "base_com",
    steps_per_iteration: int = 24,
    stage_iterations: tuple[int, int] = (1500, 3500),
    mass_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.0, 3.0),
        (0.0, 5.0),
        (0.0, 8.0),
    ),
    com_ranges: tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]], dict[str, tuple[float, float]]] = (
        {"x": (-0.02, 0.02), "y": (-0.015, 0.015), "z": (-0.005, 0.005)},
        {"x": (-0.04, 0.04), "y": (-0.025, 0.025), "z": (-0.01, 0.01)},
        {"x": (-0.04, 0.04), "y": (-0.025, 0.025), "z": (-0.01, 0.01)},
    ),
    terrain_level_caps: tuple[int, int, int] = (2, 4, 5),
) -> dict[str, torch.Tensor]:
    """Stage payload/domain randomization based on PPO training iterations."""
    iteration = env.common_step_counter // steps_per_iteration
    if iteration < stage_iterations[0]:
        stage = 0
    elif iteration < stage_iterations[1]:
        stage = 1
    else:
        stage = 2

    mass_cfg = env.event_manager.get_term_cfg(mass_event_name)
    mass_cfg.params["mass_distribution_params"] = mass_ranges[stage]

    com_cfg = env.event_manager.get_term_cfg(com_event_name)
    com_cfg.params["com_range"] = com_ranges[stage]

    terrain = env.scene.terrain
    terrain_cap = terrain_level_caps[stage]
    if getattr(terrain, "terrain_origins", None) is not None:
        max_rows = terrain.terrain_origins.shape[0]
        terrain_cap = min(terrain_cap, max_rows - 1)
        terrain.max_terrain_level = terrain_cap + 1
        terrain.terrain_levels[:] = torch.clamp(terrain.terrain_levels, 0, terrain_cap)
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]

    return {
        "stage": torch.tensor(stage + 1, device=env.device),
        "iteration": torch.tensor(iteration, device=env.device),
        "mass_max": torch.tensor(mass_ranges[stage][1], device=env.device),
        "terrain_level_cap": torch.tensor(terrain_cap, device=env.device),
    }
