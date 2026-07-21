import copy

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import CurriculumCfg, ObservationsCfg, PayloadDREventCfg, RobotEnvCfg


@configclass
class RoughObservationsCfg(ObservationsCfg):
    """Observation specifications for rough-terrain velocity tracking."""

    policy: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class RoughRobotEnvCfg(RobotEnvCfg):
    """Go2 velocity-tracking environment on Isaac Lab rough terrains."""

    observations: RoughObservationsCfg = RoughObservationsCfg()

    def __post_init__(self):
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = copy.deepcopy(ROUGH_TERRAINS_CFG)
        self.scene.terrain.max_init_terrain_level = 5

        super().__post_init__()

        # Scale down the default rough terrains for the smaller Go2 body.
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01


@configclass
class RoughRobotPlayEnvCfg(RoughRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class PayloadDRCurriculumCfg(CurriculumCfg):
    """Curriculum for progressively increasing payload/domain randomization."""

    payload_dr_levels = CurrTerm(
        func=mdp.payload_dr_levels,
        params={
            "steps_per_iteration": 24,
            "stage_iterations": (1500, 3500),
            "mass_ranges": ((0.0, 3.0), (0.0, 5.0), (0.0, 8.0)),
            "com_ranges": (
                {"x": (-0.02, 0.02), "y": (-0.015, 0.015), "z": (-0.005, 0.005)},
                {"x": (-0.04, 0.04), "y": (-0.025, 0.025), "z": (-0.01, 0.01)},
                {"x": (-0.04, 0.04), "y": (-0.025, 0.025), "z": (-0.01, 0.01)},
            ),
            "terrain_level_caps": (2, 4, 5),
        },
    )


@configclass
class RoughPayloadDRRobotEnvCfg(RoughRobotEnvCfg):
    """Go2 rough-terrain velocity task with staged payload/domain randomization."""

    events: PayloadDREventCfg = PayloadDREventCfg()
    curriculum: PayloadDRCurriculumCfg = PayloadDRCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 2


@configclass
class RoughPayloadDRRobotPlayEnvCfg(RoughPayloadDRRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
