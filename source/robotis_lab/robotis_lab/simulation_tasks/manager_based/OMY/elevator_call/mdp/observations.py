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


def call_button_depression(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Depression magnitude of CallBtn_0 from its rest position (m)."""
    pos_w, _ = _read_button_pose_world(env)  # (num_envs, 3)
    if not hasattr(env, "button_rest_pos_w"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    diff = pos_w - env.button_rest_pos_w
    return torch.linalg.vector_norm(diff, dim=-1, keepdim=True)


def call_button_pressed_physically(
    env: "ManagerBasedRLEnv",
    press_threshold: float = 0.005,
) -> torch.Tensor:
    """Success: CallBtn_0 depressed by at least press_threshold meters."""
    pos_w, _ = _read_button_pose_world(env)
    if not hasattr(env, "button_rest_pos_w"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    diff = pos_w - env.button_rest_pos_w
    return torch.linalg.vector_norm(diff, dim=-1) > press_threshold


def call_button_contact_force(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Contact force magnitude between fingertip (rh_p12_rn_r2) and CallBtn_0."""
    sensor = env.scene["btn_contact"]
    # force_matrix_w: (num_envs, num_bodies=1, num_filter_prims=1, 3)
    f = sensor.data.force_matrix_w[:, 0, 0]  # (num_envs, 3)
    return torch.linalg.vector_norm(f, dim=-1, keepdim=True)


def call_button_contact_detected(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 0.5,
) -> torch.Tensor:
    """Success when fingertip-button contact force exceeds threshold (N)."""
    sensor = env.scene["btn_contact"]
    f = sensor.data.force_matrix_w[:, 0, 0]
    mag = torch.linalg.vector_norm(f, dim=-1)
    return mag > force_threshold


def call_button_lit(env):
    """Return call button LED state (N_envs, 1). 1.0 if lit, 0.0 otherwise.

    Reads env._fsm_elevator.button.lit (set by FSM when press_call_button fires).
    """
    import torch as _t
    n = env.num_envs
    out = _t.zeros((n, 1), device=env.device, dtype=_t.float32)
    fsm = getattr(env, "_fsm_elevator", None)
    if fsm is None:
        return out
    for i in range(n):
        infos = fsm.button.btn_infos[i] if fsm.button.btn_infos else []
        calls = [idx for idx, info in enumerate(infos) if info["kind"] == "call"]
        if not calls:
            continue
        target = calls[0]
        if fsm.button.lit[i][target]:
            out[i, 0] = 1.0
    return out
