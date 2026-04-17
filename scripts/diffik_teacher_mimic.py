# Copyright 2026 Hwang Yeeun
# SPDX-License-Identifier: Apache-2.0

"""diffik-style teacher for RobotisLab-CallButton-Right-OMY-IK-Rel-Mimic-v0.

Same algorithm as diffik_teacher_jointpos.py (IKPy multi-seed → smoothstep
→ DLS micro-thrust) but outputs IK-Rel actions (delta pose in world/base frame)
via env.step(). This matches the Mimic env's action space so the demos are
directly usable by annotate_demos / generate_dataset.
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="RobotisLab-CallButton-Right-OMY-IK-Rel-Mimic-v0")
parser.add_argument("--usd-index", type=int, default=-1)
parser.add_argument("--dataset-file", type=str, default=None)
parser.add_argument("--urdf", default="/isaac-sim/output/omy_f3m/omy_f3m_abs.urdf")
parser.add_argument("--pre-offset", type=float, default=0.20)
parser.add_argument("--max-pre-press-x", type=float, default=0.30)
parser.add_argument("--press-depth", type=float, default=0.03)
parser.add_argument("--approach-steps", type=int, default=300)
parser.add_argument("--press-steps", type=int, default=1500)
parser.add_argument("--hold-steps", type=int, default=60)
parser.add_argument("--return-steps", type=int, default=200)
parser.add_argument("--contact-threshold", type=float, default=0.03)
parser.add_argument("--stall-steps", type=int, default=60)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import torch
import gymnasium as gym
from ikpy.chain import Chain

import isaaclab.sim as sim_utils
import robotis_lab  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINTS = ["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"]
TIP_LINK = "rh_p12_rn_r2"
IKPY_URDF_PATH = "/tmp/omy_arm_only.urdf"
USD_IDS = [1, 19, 22, 35, 42, 46, 81, 85, 88, 92]
USD_DIR = "/workspace/robotis_lab/third_party/elevator_setup"
IK_ACTION_SCALE = 0.1  # matches DifferentialInverseKinematicsActionCfg.scale


# ------- IKPy helpers (copied from joint_pos teacher) -------
def _build_arm_urdf(src, dst):
    tree = ET.parse(src); root = tree.getroot()
    keep_links = {"world", "link0", "link1", "link2", "link3", "link4", "link5", "link6", "end_effector_flange_link"}
    keep_joints = {"world_fixed", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "end_effector_flange_joint"}
    for e in list(root):
        if e.tag == "link" and e.get("name") not in keep_links:
            root.remove(e)
        elif e.tag == "joint" and e.get("name") not in keep_joints:
            root.remove(e)
    tree.write(dst)

_build_arm_urdf(args.urdf, IKPY_URDF_PATH)
_chain = Chain.from_urdf_file(IKPY_URDF_PATH, base_elements=["link0"],
    active_links_mask=[False, True, True, True, True, True, True, False])

def _make_q(arm6):
    q = np.zeros(8, dtype=np.float64); q[1:7] = arm6; return q

def fk(arm6):
    return _chain.forward_kinematics(_make_q(arm6))  # 4x4 base-frame

def _solve_ik(target_pos, target_rot, seed_q, n_seeds=20):
    best_q, best_err = None, 1e9
    base = seed_q.copy()
    for s in range(n_seeds):
        init = _make_q(base if s == 0 else base + np.random.uniform(-0.2, 0.2, 6))
        ik = _chain.inverse_kinematics(target_position=target_pos, target_orientation=target_rot,
            orientation_mode="all", initial_position=init)
        q = ik[1:7]
        f = _chain.forward_kinematics(_make_q(q))
        err = float(np.linalg.norm(f[:3, 3] - target_pos))
        if err < best_err: best_err, best_q = err, q
        if err < 1e-3: break
    return best_q, best_err


# ------- Frame helpers -------
def yaw_rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def world_to_base(p_w, root_pos_w, root_yaw):
    return yaw_rot(root_yaw).T @ (p_w - root_pos_w)

def base_to_world(p_b, root_pos_w, root_yaw):
    return root_pos_w + yaw_rot(root_yaw) @ p_b

def rotmat_to_axis_angle(R):
    """Rotation matrix (3x3) → axis-angle (3,)."""
    cos_th = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(cos_th)
    if abs(th) < 1e-7:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(th))
    return axis * th

def quat_to_rotmat_wxyz(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),   1 - 2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),     1 - 2*(x*x+y*y)],
    ])


# ------- Env setup -------
env_cfg = parse_env_cfg(args.task, num_envs=1)
env_cfg.seed = args.seed
if 0 <= args.usd_index < len(USD_IDS):
    env_cfg.scene.elevator.spawn = sim_utils.UsdFileCfg(
        usd_path=f"{USD_DIR}/elevator_setup_{USD_IDS[args.usd_index]:03d}.usd"
    )
    print(f"[SETUP] pinned USD_{USD_IDS[args.usd_index]:03d}")
env_cfg.episode_length_s = 30.0

if args.dataset_file:
    output_dir = os.path.dirname(args.dataset_file) or "."
    os.makedirs(output_dir, exist_ok=True)
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args.dataset_file))[0]
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    print(f"[SETUP] will export on success → {args.dataset_file}")

env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

robot = env.scene["robot"]
device = env.device
arm_ids = robot.find_joints(ARM_JOINTS)[0]


# ------- Read scene state -------
from robotis_lab.simulation_tasks.manager_based.OMY.elevator_call.mdp.observations import (
    _read_button_pose_world,
)

btn_pos_w, _ = _read_button_pose_world(env)
button_w = btn_pos_w[0].cpu().numpy().astype(np.float64)

robot_pos_w = robot.data.root_pos_w[0].cpu().numpy().astype(np.float64)
robot_quat_w = robot.data.root_quat_w[0].cpu().numpy().astype(np.float64)
robot_yaw = float(2.0 * np.arctan2(robot_quat_w[3], robot_quat_w[0]))

button_b = world_to_base(button_w, robot_pos_w, robot_yaw)
pre_press_x_ideal = float(button_b[0]) - args.pre_offset
pre_press_x = min(pre_press_x_ideal, args.max_pre_press_x)
pre_press_b = np.array([pre_press_x, float(button_b[1]), float(button_b[2])])
press_b = button_b + np.array([args.press_depth, 0.0, 0.0])

# R_face_button (base frame)
_press_dir_b = np.array([1.0, 0.0, 0.0]); _world_up_b = np.array([0.0, 0.0, 1.0])
fy = -_press_dir_b; fz = _world_up_b - fy*np.dot(_world_up_b, fy); fz /= np.linalg.norm(fz); fx = np.cross(fy, fz)
R_face_button = np.column_stack([fx, fy, fz])

home_arm = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float64)
goal_pre_q, pre_err = _solve_ik(pre_press_b, R_face_button, home_arm)
print(f"[SETUP] goal_pre_q={np.round(goal_pre_q, 3).tolist()} ik_err={pre_err:.4f}m")


# ------- Action helpers -------
def eef_world_from_obs():
    """Read current EEF world (pos, rotmat) from env observation."""
    pos_w = env.scene["ee_frame"].data.target_pos_w[0, 0].detach().cpu().numpy().astype(np.float64)
    quat_w = env.scene["ee_frame"].data.target_quat_w[0, 0].detach().cpu().numpy().astype(np.float64)
    return pos_w, quat_to_rotmat_wxyz(quat_w)


def make_ik_rel_action_from_target_eef(target_pos_world, target_rot_world, gripper=1.0):
    """Compute IK-Rel action = (delta / IK_ACTION_SCALE), clamped to [-1,1]."""
    curr_pos_w, curr_rot_w = eef_world_from_obs()
    delta_pos = target_pos_world - curr_pos_w
    delta_R = target_rot_world @ curr_rot_w.T
    delta_aa = rotmat_to_axis_angle(delta_R)

    action_pos = np.clip(delta_pos / IK_ACTION_SCALE, -1.0, 1.0)
    action_rot = np.clip(delta_aa / IK_ACTION_SCALE, -1.0, 1.0)
    vec = np.concatenate([action_pos, action_rot, [gripper]]).astype(np.float32)
    return torch.tensor(vec, device=device).unsqueeze(0)


def joint_target_to_world_eef(arm_q):
    """Convert target arm joint config to (target_pos_w, target_rot_w)."""
    M_b = fk(arm_q)
    target_pos_b = M_b[:3, 3]
    target_rot_b = M_b[:3, :3]
    target_pos_w = base_to_world(target_pos_b, robot_pos_w, robot_yaw)
    target_rot_w = yaw_rot(robot_yaw) @ target_rot_b
    return target_pos_w, target_rot_w


def step(action):
    return env.step(action)


def did_term(term, trunc):
    as_bool = lambda x: bool(x.any().item() if hasattr(x, "any") and hasattr(x.any(), "item") else (x.any() if hasattr(x, "any") else x))
    return as_bool(term) or as_bool(trunc)


# ------- Phase 1: Settle (zero delta, gripper open for this task) -------
print("[PHASE] settle")
curr_pos_w, curr_rot_w = eef_world_from_obs()
settle_action = make_ik_rel_action_from_target_eef(curr_pos_w, curr_rot_w, gripper=1.0)
for _ in range(30):
    obs, _, term, trunc, _ = step(settle_action)
    if did_term(term, trunc): break


# ------- Phase 2: Approach — smoothstep through joint configs, each step command EEF pose -------
print(f"[PHASE] approach: {args.approach_steps} steps")
start_q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float64)
terminated_early = False
for i in range(args.approach_steps):
    t = i / max(args.approach_steps - 1, 1)
    alpha = t * t * (3.0 - 2.0 * t)
    q_cmd = (1.0 - alpha) * start_q + alpha * goal_pre_q
    target_pos_w, target_rot_w = joint_target_to_world_eef(q_cmd)
    action = make_ik_rel_action_from_target_eef(target_pos_w, target_rot_w, gripper=1.0)
    obs, _, term, trunc, _ = step(action)
    if (i + 1) % 60 == 0:
        tip_w = robot.data.body_pos_w[0, robot.find_bodies(TIP_LINK)[0][0]].detach().cpu().numpy()
        print(f"  approach step={i+1:3d}/{args.approach_steps} alpha={alpha:.3f} tip_dist={np.linalg.norm(tip_w - button_w):.3f}m")
    if did_term(term, trunc):
        terminated_early = True
        break


# ------- Phase 3: Press — EEF target at press_world, small incremental commands -------
pressed = False
min_dist = float("inf")
if not terminated_early:
    print(f"[PHASE] press (direct IK-Rel to target): up to {args.press_steps} steps")
    press_pos_w = base_to_world(press_b, robot_pos_w, robot_yaw)
    # Maintain EEF orientation from end of approach
    _, press_rot_w = joint_target_to_world_eef(goal_pre_q)

    stall = 0
    tip_frame_idx = robot.find_bodies(TIP_LINK)[0][0]
    for i in range(args.press_steps):
        action = make_ik_rel_action_from_target_eef(press_pos_w, press_rot_w, gripper=1.0)
        obs, _, term, trunc, _ = step(action)
        tip_w = robot.data.body_pos_w[0, tip_frame_idx].detach().cpu().numpy()
        dist = float(np.linalg.norm(tip_w - button_w))
        if dist < min_dist - 1e-4:
            min_dist = dist; stall = 0
        else:
            stall += 1
        if (i + 1) % 100 == 0:
            print(f"  press step={i+1:4d} dist={dist:.4f}m stall={stall}")
        if dist <= args.contact_threshold:
            print(f"  [PRESSED] contact at step {i+1} dist={dist:.4f}m")
            pressed = True; break
        if stall >= args.stall_steps:
            print(f"  [STALLED] no progress for {stall} steps, dist={dist:.4f}m"); break
        if did_term(term, trunc):
            pressed = True; break


# ------- Phase 4+5: hold + retract -------
if not terminated_early:
    print("[PHASE] hold")
    press_pos_w = base_to_world(press_b, robot_pos_w, robot_yaw)
    _, press_rot_w = joint_target_to_world_eef(goal_pre_q)
    for _ in range(args.hold_steps):
        obs, _, term, trunc, _ = step(make_ik_rel_action_from_target_eef(press_pos_w, press_rot_w, gripper=1.0))

    print(f"[PHASE] retract: {args.return_steps} steps")
    pre_press_w = base_to_world(pre_press_b, robot_pos_w, robot_yaw)
    for _ in range(args.return_steps):
        obs, _, term, trunc, _ = step(make_ik_rel_action_from_target_eef(pre_press_w, press_rot_w, gripper=1.0))


# ------- Summary + export -------
tip_w_final = robot.data.body_pos_w[0, robot.find_bodies(TIP_LINK)[0][0]].detach().cpu().numpy()
final_dist = float(np.linalg.norm(tip_w_final - button_w))
print("\n" + "=" * 60)
if 0 <= args.usd_index < len(USD_IDS):
    print(f"[RESULT] USD_{USD_IDS[args.usd_index]:03d}")
print(f"[RESULT] pressed      : {pressed}")
print(f"[RESULT] min_distance : {min_dist:.4f} m")
print(f"[RESULT] final_dist   : {final_dist:.4f} m")

if args.dataset_file:
    try:
        for env_id, ep in getattr(env.recorder_manager, "_episodes", {}).items():
            if ep is not None and not ep.is_empty():
                ep.success = bool(pressed)
        env.recorder_manager.cfg.dataset_export_mode = (
            DatasetExportMode.EXPORT_ALL if pressed else DatasetExportMode.EXPORT_NONE
        )
        env.recorder_manager.export_episodes()
        print(f"[RESULT] exported to  : {args.dataset_file}" if pressed
              else f"[RESULT] NOT exported : pressed=False")
    except Exception as e:
        print(f"[RESULT] export FAILED : {e}")
print("=" * 60)
env.close()
