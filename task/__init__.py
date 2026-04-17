# Copyright 2026 Hwang Yeeun
# SPDX-License-Identifier: Apache-2.0

import os

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotisLab-CallButton-Right-OMY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OMYElevatorCallEnvCfg",
        "robomimic_bc_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_image.json"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="RobotisLab-CallButton-Right-OMY-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:OMYElevatorCallEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="RobotisLab-CallButton-Right-OMY-IK-Rel-Mimic-v0",
    entry_point=f"{__name__}.mimic_env:OMYElevatorCallMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:OMYElevatorCallMimicEnvCfg",
    },
    disable_env_checker=True,
)
