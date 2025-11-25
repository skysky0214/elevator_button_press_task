# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv, ManagerBasedRLEnvCfg


class FFWSG2PickPlaceMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for FFW SG2 Pick and Place.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_root_pos = self.scene['robot'].data.root_pos_w
        self.robot_root_quat = self.scene['robot'].data.root_quat_w

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)

        # For FFW SG2, if eef_name is the robot name (e.g., "FFW_SG2"),
        # return the right EEF pose as the primary manipulator
        # Otherwise, try to get the specific EEF state from observation buffer
        
        # Map robot name to primary EEF (right arm for FFW SG2)
        if eef_name in ["left_ee_frame_state", "right_ee_frame_state"]:
            eef_state_key = eef_name
        else:
            # Try using eef_name directly
            eef_state_key = eef_name
        
        if eef_state_key not in self.obs_buf["policy"]:
            raise ValueError(
                f"EEF state key '{eef_state_key}' (from eef_name '{eef_name}') "
                f"not found in observation buffer. Available keys: {list(self.obs_buf['policy'].keys())}"
            )
        
        eef_state = self.obs_buf["policy"][eef_state_key][env_ids]
        eef_pos = eef_state[:, :3]
        eef_quat = eef_state[:, 3:7]
        # quat: (w, x, y, z)
        eef_pose = PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

        return eef_pose

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        # FFW SG2 dual-arm support: extract left and right EEF poses
        # Expected keys: "left_ee_frame_state" and "right_ee_frame_state"
        left_eef_pose = target_eef_pose_dict.get("left_ee_frame_state")
        right_eef_pose = target_eef_pose_dict.get("right_ee_frame_state")
        
        if left_eef_pose is None or right_eef_pose is None:
            raise ValueError(
                f"Expected 'left_ee_frame_state' and 'right_ee_frame_state' in target_eef_pose_dict, "
                f"got keys: {list(target_eef_pose_dict.keys())}"
            )
        
        # Convert left EEF pose to pos + quat
        left_eef_pos, left_eef_rot = PoseUtils.unmake_pose(left_eef_pose)
        left_eef_quat = PoseUtils.quat_from_matrix(left_eef_rot)
        left_pose_action = torch.cat([left_eef_pos, left_eef_quat], dim=0)
        
        # Convert right EEF pose to pos + quat
        right_eef_pos, right_eef_rot = PoseUtils.unmake_pose(right_eef_pose)
        right_eef_quat = PoseUtils.quat_from_matrix(right_eef_rot)
        right_pose_action = torch.cat([right_eef_pos, right_eef_quat], dim=0)
        
        # Extract gripper actions for both arms
        left_gripper = gripper_action_dict.get("left_ee_frame_state")
        right_gripper = gripper_action_dict.get("right_ee_frame_state")
        
        if left_gripper is None or right_gripper is None:
            raise ValueError(
                f"Expected 'left_ee_frame_state' and 'right_ee_frame_state' gripper actions, "
                f"got keys: {list(gripper_action_dict.keys())}"
            )

        # Get current lift and head positions from observation
        # These are not controlled by IK, so we keep their current values
        joint_pos = self.obs_buf["policy"]["joint_pos"][env_id]
        lift_action = joint_pos[16:17]    # Index 16: lift_joint
        head_action = joint_pos[17:19]    # Index 17-18: head_joint[1-2]

        # Concatenate: [left_eef(7), right_eef(7), gripper_l(1), gripper_r(1), lift(1), head(2)] = 20D
        action = torch.cat([
            left_pose_action,   # 0-6
            right_pose_action,  # 7-13
            left_gripper,       # 14
            right_gripper,      # 15
            lift_action,        # 16
            head_action         # 17-18
        ], dim=0)
        
        return action.unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        # FFW SG2 dual-arm: action format = [left_eef(7), right_eef(7), gripper_l(1), gripper_r(1), lift(1), head(2)]
        # Extract left EEF (first 7 dimensions: pos + quat)
        left_eef_pos = action[:, :3]
        left_eef_quat = action[:, 3:7]
        left_eef_rot = PoseUtils.matrix_from_quat(left_eef_quat)
        left_eef_pose = PoseUtils.make_pose(left_eef_pos, left_eef_rot).clone()
        
        # Extract right EEF (next 7 dimensions: pos + quat)
        right_eef_pos = action[:, 7:10]
        right_eef_quat = action[:, 10:14]
        right_eef_rot = PoseUtils.matrix_from_quat(right_eef_quat)
        right_eef_pose = PoseUtils.make_pose(right_eef_pos, right_eef_rot).clone()

        return {
            "left_ee_frame_state": left_eef_pose,
            "right_ee_frame_state": right_eef_pose
        }

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        # FFW SG2 dual-arm: extract both gripper actions
        # Action format: [left_eef(7), right_eef(7), gripper_l(1), gripper_r(1), lift(1), head(2)]
        left_gripper = actions[:, 14:15]   # Index 14: left gripper
        right_gripper = actions[:, 15:16]  # Index 15: right gripper
        
        return {
            "left_ee_frame_state": left_gripper,
            "right_ee_frame_state": right_gripper
        }

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. The implementation of this method is
        required if intending to enable automatic subtask term signal annotation when running the
        dataset annotation tool. This method can be kept unimplemented if intending to use manual
        subtask term signal annotation.

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        for term_name, term_signal in subtask_terms.items():
            signals[term_name] = term_signal[env_ids]

        return signals
