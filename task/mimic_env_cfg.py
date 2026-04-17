# Copyright 2026 Hwang Yeeun
# SPDX-License-Identifier: Apache-2.0

# Mimic env config for OMY elevator call-button press (IK-Rel).
# Mirrors the cabinet Mimic wiring.

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robotis_lab.simulation_tasks.manager_based.OMY.elevator_call.mdp as mdp
from .ik_rel_env_cfg import OMYElevatorCallEnvCfg


@configclass
class _SubtaskCfg(ObsGroup):
    """Subtask termination signals consumed by Mimic auto-annotation."""

    press_done = ObsTerm(
        func=mdp.call_button_pressed,
        params={"distance_threshold": 0.03},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class OMYElevatorCallMimicEnvCfg(OMYElevatorCallEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Success termination is already defined in base ElevatorCallEnvCfg.

        # Mimic reads these from obs_buf["policy"] and obs_buf["subtask_terms"].
        self.observations.policy.eef_pos = ObsTerm(func=mdp.ee_pos)
        self.observations.policy.eef_quat = ObsTerm(func=mdp.ee_quat)
        self.observations.policy.concatenate_terms = False
        self.observations.subtask_terms = _SubtaskCfg()

        # --- Datagen config ---
        self.datagen_config.name = "demo_src_callbutton_omy_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # Single subtask: reach and press CallBtn_0. Button position is the
        # only object-level variation across USDs + xy randomization.
        self.subtask_configs["omy"] = [
            SubTaskConfig(
                object_ref="call_button",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Reach and press the hall call button",
            ),
        ]
