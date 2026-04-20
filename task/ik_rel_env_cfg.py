# IK-relative action variant. Required for the Mimic pipeline.

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

from robotis_lab.assets.robots.OMY import OMY_CFG  # isort: skip


@configclass
class OMYElevatorCallEnvCfg(joint_pos_env_cfg.OMYElevatorCallEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Reapply OMY with the same init_state (so it stays on the pedestal).
        self.scene.robot = self.scene.robot

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            body_name="link6",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.1,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )
