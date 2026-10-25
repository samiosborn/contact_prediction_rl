"""Small initial actions for snapshot-backed finger-only Reorientation."""

from isaaclab.utils import configclass

from .agents.ppo_cfg import PPORunnerCfg


@configclass
class ReorientationTrainPPORunnerCfg(PPORunnerCfg):
    def __post_init__(self) -> None:
        self.actor["initial_std"] = 0.2
        self.actor["action_head_init_scale"] = 0.01
        self.algorithm["learning_rate"] = 1.0e-4
        self.experiment_name = "hx5_d20_reorientation_train"
