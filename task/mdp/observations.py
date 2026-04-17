# Copyright 2026 Hwang Yeeun
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.sensors import FrameTransformer, FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_CALL_BUTTON_PRIM_SUFFIX = "/Elevator/HallExterior/CallBtn_0"


def _read_button_pose_world(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """Read CallBtn_0 world (pos, quat) per env directly from USD xforms.

    Bypasses FrameTransformer / Articulation because the elevator USDs'
    PrismaticJoints have body0=[] with mismatched anchors, which causes PhysX
    to "snap" the rigid button bodies away from their USD placement.
    """
    import omni.usd
    from pxr import UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()
    device = env.device
    n = env.num_envs

    positions = torch.zeros((n, 3), device=device)
    quats = torch.zeros((n, 4), device=device)  # w,x,y,z
    for env_id in range(n):
        path = f"/World/envs/env_{env_id}/Elevator{_CALL_BUTTON_PRIM_SUFFIX}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        t = xform.ExtractTranslation()
        r = xform.ExtractRotationQuat()  # Gf.Quatf, real, imag
        positions[env_id] = torch.tensor([t[0], t[1], t[2]], device=device)
        real = r.GetReal()
        imag = r.GetImaginary()
        quats[env_id] = torch.tensor([real, imag[0], imag[1], imag[2]], device=device)
    return positions, quats


def ee_pos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """EEF position in the environment frame."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    return ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins


def ee_quat(env: "ManagerBasedRLEnv", make_quat_unique: bool = True) -> torch.Tensor:
    """EEF orientation."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    q = ee_tf_data.target_quat_w[..., 0, :]
    return math_utils.quat_unique(q) if make_quat_unique else q


def call_button_pos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """CallBtn_0 position in the environment frame."""
    pos_w, _ = _read_button_pose_world(env)
    return pos_w - env.scene.env_origins


def call_button_quat(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """CallBtn_0 orientation (world frame; env rotation is identity here)."""
    _, quat_w = _read_button_pose_world(env)
    return quat_w


def rel_ee_call_button_distance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Vector from EEF to CallBtn_0, world frame (env offset cancels)."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    pos_w, _ = _read_button_pose_world(env)
    return pos_w - ee_tf_data.target_pos_w[..., 0, :]


def call_button_pressed(
    env: "ManagerBasedRLEnv",
    distance_threshold: float = 0.03,
) -> torch.Tensor:
    """Subtask / success signal: True when EEF is within `distance_threshold` of CallBtn_0."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    pos_w, _ = _read_button_pose_world(env)
    diff = pos_w - ee_tf_data.target_pos_w[..., 0, :]
    return torch.linalg.vector_norm(diff, dim=-1) < distance_threshold
