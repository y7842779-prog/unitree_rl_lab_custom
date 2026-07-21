from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_com_from_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize body CoM from the initial CoM instead of accumulating offsets."""
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    coms = asset.root_physx_view.get_coms().clone()
    default_attr = "_unitree_rl_lab_default_coms"
    if not hasattr(asset, default_attr):
        setattr(asset, default_attr, coms.clone())
    default_coms = getattr(asset, default_attr)

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu"
    ).unsqueeze(1)

    randomized_coms = default_coms[env_ids[:, None], body_ids].clone()
    randomized_coms[..., :3] += rand_samples
    coms[env_ids[:, None], body_ids] = randomized_coms
    asset.root_physx_view.set_coms(coms, env_ids)
