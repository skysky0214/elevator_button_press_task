# Copyright 2026 Hwang Yeeun
# SPDX-License-Identifier: Apache-2.0

"""
diffik-style scripted teacher for RobotisLab-CallButton-Right-OMY-v0 (joint_pos env).

Port of /root/isaac_sim_demo/example/omy_f3m_press_diffik.py, adapted to run
inside Isaac Lab's env.step() loop so the action stream can be recorded to
HDF5 for imitation learning.

Phases:
  settle    — keep gripper closed and let physics stabilize
  approach  — joint-space cubic smoothstep from HOME to goal_pre_q (IKPy solved)
  press     — DLS micro-thrust on the gripper tip body (0.6 mm/step)
  hold      — keep final config for a few steps
  retract   — smoothstep back to goal_pre_q

Action convention:
  arm_action = JointPositionActionCfg(use_default_offset=True, scale=1.0)
    → action_arm = target_joint_pos - default_joint_pos
  gripper_action = BinaryJointPositionActionCfg
    → action_gripper = +1 (closed) throughout

Usage (livestream preview):
  /isaac-sim/python.sh /tmp/diffik_teacher_jointpos.py --usd-index 0 --livestream 2

Usage (headless, record):
  /isaac-sim/python.sh /tmp/diffik_teacher_jointpos.py --usd-index 0 --headless \
      --dataset-file /isaac-sim/output/callbutton_demo_0.hdf5
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="RobotisLab-CallButton-Right-OMY-v0")
parser.add_argument("--usd-index", type=int, default=-1,
                    help="0..9: pin to a specific elevator USD; -1 = MultiAsset random")
parser.add_argument("--dataset-file", type=str, default=None)
parser.add_argument("--urdf", default="/isaac-sim/output/omy_f3m/omy_f3m_abs.urdf")
parser.add_argument("--pre-offset", type=float, default=0.20,
                    help="pre-press offset (m) from button along approach axis (base frame)")
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
from isaaclab.managers import DatasetExportMode


# Robot / URDF constants (match diffik script)
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINTS = ["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"]
EE_LINK = "end_effector_flange_link"
TIP_LINK = "rh_p12_rn_r2"
IKPY_URDF_PATH = "/tmp/omy_arm_only.urdf"

USD_IDS = [1, 19, 22, 35, 42, 46, 81, 85, 88, 92]
USD_DIR = "/workspace/robotis_lab/third_party/elevator_setup"


# -----------------------------------------------------------------------------
# IKPy arm-only chain builder + multi-seed solver (copied from diffik)
# -----------------------------------------------------------------------------
def _build_arm_urdf(src, dst):
    tree = ET.parse(src)
    root = tree.getroot()
    keep_links = {"world", "link0", "link1", "link2", "link3", "link4", "link5", "link6", "end_effector_flange_link"}
    keep_joints = {"world_fixed", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "end_effector_flange_joint"}
    for e in list(root):
        if e.tag == "link" and e.get("name") not in keep_links:
            root.remove(e)
        elif e.tag == "joint" and e.get("name") not in keep_joints:
            root.remove(e)
    tree.write(dst)


_build_arm_urdf(args.urdf, IKPY_URDF_PATH)
_chain = Chain.from_urdf_file(
    IKPY_URDF_PATH,
    base_elements=["link0"],
    active_links_mask=[False, True, True, True, True, True, True, False],
)


def _make_q(arm6):
    q = np.zeros(8, dtype=np.float64)
    q[1:7] = arm6
    return q


def _solve_ik(target_pos, target_rot, seed_q, n_seeds=20):
    best_q, best_err = None, 1e9
    base = seed_q.copy()
    for s in range(n_seeds):
        init = _make_q(base if s == 0 else base + np.random.uniform(-0.2, 0.2, 6))
        ik = _chain.inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_rot,
            orientation_mode="all",
            initial_position=init,
        )
        q = ik[1:7]
        fk = _chain.forward_kinematics(_make_q(q))
        err = float(np.linalg.norm(fk[:3, 3] - target_pos))
        if err < best_err:
            best_err, best_q = err, q
        if err < 1e-3:
            break
    return best_q, best_err


def world_to_base(p_w, root_pos_w, root_yaw):
    c, s = np.cos(root_yaw), np.sin(root_yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R.T @ (p_w - root_pos_w)


# -----------------------------------------------------------------------------
# Env setup
# -----------------------------------------------------------------------------
env_cfg = parse_env_cfg(args.task, num_envs=1)
env_cfg.seed = args.seed

if 0 <= args.usd_index < len(USD_IDS):
    env_cfg.scene.elevator.spawn = sim_utils.UsdFileCfg(
        usd_path=f"{USD_DIR}/elevator_setup_{USD_IDS[args.usd_index]:03d}.usd"
    )
    print(f"[SETUP] pinned USD_{USD_IDS[args.usd_index]:03d}")

# Extend episode so the full trajectory fits.
env_cfg.episode_length_s = 30.0

# Recording setup — swap the default (empty) RecorderManagerBaseCfg for
# ActionStateRecorderManagerCfg, which bundles initial_state / actions / obs
# terms. Without these active terms, export_episodes() returns silently and
# no HDF5 is written.
if args.dataset_file:
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
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
grip_ids = robot.find_joints(GRIPPER_JOINTS)[0]

# Gripper: leave open. patch_gripper_drives is disabled, so mimic stays bound
# and fighting it has no benefit. The demos record the arm motion correctly.
GRIPPER_CLOSED = 1.0  # still used by action: keep action=close so policy learns "always close".


def _enforce_gripper_closed(teleport: bool = False):
    return


# -----------------------------------------------------------------------------
# Read button/robot pose, compute targets in base frame
# -----------------------------------------------------------------------------
from robotis_lab.simulation_tasks.manager_based.OMY.elevator_call.mdp.observations import (
    _read_button_pose_world,
)

button_pos_w, _ = _read_button_pose_world(env)
button_w = button_pos_w[0].cpu().numpy().astype(np.float64)

robot_pos_w = robot.data.root_pos_w[0].cpu().numpy().astype(np.float64)
robot_quat_w = robot.data.root_quat_w[0].cpu().numpy().astype(np.float64)  # (w,x,y,z)
# Extract yaw from quaternion (assumes rotation only about Z)
robot_yaw = float(2.0 * np.arctan2(robot_quat_w[3], robot_quat_w[0]))

button_base = world_to_base(button_w, robot_pos_w, robot_yaw)

# pre-press target: buffered from button along -x base, capped
pre_press_x_ideal = float(button_base[0]) - args.pre_offset
pre_press_x = min(pre_press_x_ideal, args.max_pre_press_x)
pre_press_b = np.array([pre_press_x, float(button_base[1]), float(button_base[2])])
press_b = button_base + np.array([args.press_depth, 0.0, 0.0])

# Face-button orientation (flange -Y aligned with +x press direction)
_press_dir_b = np.array([1.0, 0.0, 0.0])
_world_up_b = np.array([0.0, 0.0, 1.0])
fy = -_press_dir_b
fz = _world_up_b - fy * np.dot(_world_up_b, fy)
fz /= np.linalg.norm(fz)
fx = np.cross(fy, fz)
R_face_button = np.column_stack([fx, fy, fz])

# HOME arm config
home_arm = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float64)

goal_pre_q, pre_err = _solve_ik(pre_press_b, R_face_button, home_arm)
print(f"[SETUP] button_w      = {np.round(button_w, 4).tolist()}")
print(f"[SETUP] robot_w pos   = {np.round(robot_pos_w, 4).tolist()}  yaw={robot_yaw:.3f}")
print(f"[SETUP] button_base   = {np.round(button_base, 4).tolist()}")
print(f"[SETUP] pre_press_b   = {np.round(pre_press_b, 4).tolist()}")
print(f"[SETUP] press_b       = {np.round(press_b, 4).tolist()}")
print(f"[SETUP] home_arm      = {np.round(home_arm, 3).tolist()}")
print(f"[SETUP] goal_pre_q    = {np.round(goal_pre_q, 3).tolist()}  ik_err={pre_err:.4f}m")


# -----------------------------------------------------------------------------
# Action helpers
# -----------------------------------------------------------------------------
default_arm = robot.data.default_joint_pos[0, arm_ids].cpu().numpy().astype(np.float64)


def _make_action(arm_target, gripper=1.0):
    """Build full 7-D env action from absolute arm joint targets + gripper scalar."""
    arm_delta = (arm_target - default_arm).astype(np.float32)
    gripper_arr = np.array([gripper], dtype=np.float32)
    a = np.concatenate([arm_delta, gripper_arr])
    return torch.tensor(a, device=device).unsqueeze(0)


def _did_terminate(terminated, truncated):
    def _as_bool(x):
        if hasattr(x, "any"):
            return bool(x.any().item() if hasattr(x, "item") else x.any())
        return bool(x)
    return _as_bool(terminated) or _as_bool(truncated)


# -----------------------------------------------------------------------------
# Phase 1: settle (gripper close + physics stabilization)
# -----------------------------------------------------------------------------
settle_action = _make_action(home_arm, gripper=1.0)
print("[PHASE] settle")
for i in range(30):
    obs, rew, term, trunc, info = env.step(settle_action)
    # First 20 steps: teleport gripper joints back to closed every tick.
    _enforce_gripper_closed(teleport=(i < 20))
    if _did_terminate(term, trunc):
        print(f"  early termination during settle at step {i}")
        break
_grip_actual = robot.data.joint_pos[0, grip_ids].detach().cpu().numpy()
print(f"[GRIPPER] after settle: target={GRIPPER_CLOSED} actual={np.round(_grip_actual, 3).tolist()}")


# -----------------------------------------------------------------------------
# Phase 2: Approach — joint-space cubic smoothstep from HOME to goal_pre_q
# -----------------------------------------------------------------------------
print(f"[PHASE] approach: {args.approach_steps} steps")
start_q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float64)
terminated_early = False
for i in range(args.approach_steps):
    t = i / max(args.approach_steps - 1, 1)
    alpha = t * t * (3.0 - 2.0 * t)
    q_cmd = (1.0 - alpha) * start_q + alpha * goal_pre_q
    obs, rew, term, trunc, info = env.step(_make_action(q_cmd, gripper=1.0))
    _enforce_gripper_closed()
    if (i + 1) % 60 == 0:
        tip_w = robot.data.body_pos_w[0, robot.find_bodies(TIP_LINK)[0][0]].detach().cpu().numpy()
        dist = float(np.linalg.norm(tip_w - button_w))
        print(f"  approach step={i+1:3d}/{args.approach_steps}  alpha={alpha:.3f}  tip_dist={dist:.3f}m")
    if _did_terminate(term, trunc):
        print(f"  early termination during approach at step {i}")
        terminated_early = True
        break


# -----------------------------------------------------------------------------
# Phase 3: Press — DLS micro-thrust on TIP body
# -----------------------------------------------------------------------------
pressed = False
min_dist = float("inf")
if not terminated_early:
    print(f"[PHASE] press (micro-thrust): up to {args.press_steps} steps")
    tip_frame_idx = robot.find_bodies(TIP_LINK)[0][0]
    tip_jacobi_idx = tip_frame_idx - 1

    PUSH_STEP_M = 0.0006
    JOINT_CLAMP_RAD = 0.005
    DLS_LAMBDA = 0.05

    def damped_pinv_torch(J, lam):
        JT = J.transpose(-2, -1)
        eye = torch.eye(J.shape[-2], device=J.device).unsqueeze(0)
        return JT @ torch.inverse(J @ JT + lam * lam * eye)

    button_w_t = torch.tensor([button_w], dtype=torch.float32, device=device)
    current_targets = robot.data.joint_pos[:, arm_ids].clone()
    stall = 0

    for i in range(args.press_steps):
        tip_w_tensor = robot.data.body_pos_w[:, tip_frame_idx]
        tip_to_btn = button_w_t - tip_w_tensor
        tip_to_btn_norm = tip_to_btn / (torch.norm(tip_to_btn, dim=-1, keepdim=True) + 1e-6)
        push = tip_to_btn_norm * PUSH_STEP_M

        J_lin = robot.root_physx_view.get_jacobians()[:, tip_jacobi_idx, :3, arm_ids]
        delta = (damped_pinv_torch(J_lin, DLS_LAMBDA) @ push.unsqueeze(-1)).squeeze(-1)
        delta = torch.clamp(delta, -JOINT_CLAMP_RAD, JOINT_CLAMP_RAD)
        current_targets = current_targets + delta

        arm_target_np = current_targets[0].cpu().numpy()
        obs, rew, term, trunc, info = env.step(_make_action(arm_target_np, gripper=1.0))
        _enforce_gripper_closed()

        tip_now = robot.data.body_pos_w[0, tip_frame_idx].detach().cpu().numpy()
        dist = float(np.linalg.norm(tip_now - button_w))
        if dist < min_dist - 1e-4:
            min_dist = dist
            stall = 0
        else:
            stall += 1

        if (i + 1) % 100 == 0:
            print(f"  press step={i+1:4d}  dist={dist:.4f}m  stall={stall}")

        if dist <= args.contact_threshold:
            print(f"  [PRESSED] contact at step {i+1} dist={dist:.4f}m")
            pressed = True
            break
        if stall >= args.stall_steps:
            print(f"  [STALLED] no progress for {stall} steps, dist={dist:.4f}m")
            break
        if _did_terminate(term, trunc):
            print(f"  env terminated during press at step {i+1} dist={dist:.4f}m")
            pressed = True  # env marks success via termination
            break


# -----------------------------------------------------------------------------
# Phase 4 + 5: Hold + Retract
# -----------------------------------------------------------------------------
if not terminated_early:
    print("[PHASE] hold")
    hold_target = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float64)
    for _ in range(args.hold_steps):
        obs, rew, term, trunc, info = env.step(_make_action(hold_target, gripper=1.0))
        _enforce_gripper_closed()

    print(f"[PHASE] retract: {args.return_steps} steps")
    current_q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().astype(np.float64)
    for i in range(args.return_steps):
        t = i / max(args.return_steps - 1, 1)
        alpha = t * t * (3.0 - 2.0 * t)
        q_cmd = (1.0 - alpha) * current_q + alpha * goal_pre_q
        obs, rew, term, trunc, info = env.step(_make_action(q_cmd, gripper=1.0))
    _enforce_gripper_closed()


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
tip_w_final = robot.data.body_pos_w[0, robot.find_bodies(TIP_LINK)[0][0]].detach().cpu().numpy()
final_dist = float(np.linalg.norm(tip_w_final - button_w))

print("")
print("=" * 60)
if 0 <= args.usd_index < len(USD_IDS):
    print(f"[RESULT] USD_{USD_IDS[args.usd_index]:03d}")
print(f"[RESULT] pressed      : {pressed}")
print(f"[RESULT] min_distance : {min_dist:.4f} m")
print(f"[RESULT] final_dist   : {final_dist:.4f} m")
if args.dataset_file:
    # Flush recorder_manager to HDF5 regardless of whether success termination
    # actually fired during the run. Mark buffered episode as successful iff
    # the teacher considered the demo pressed, then force export.
    try:
        from isaaclab.managers import DatasetExportMode
        for env_id, ep in getattr(env.recorder_manager, "_episodes", {}).items():
            if ep is not None and not ep.is_empty():
                ep.success = bool(pressed)
        env.recorder_manager.cfg.dataset_export_mode = (
            DatasetExportMode.EXPORT_ALL if pressed else DatasetExportMode.EXPORT_NONE
        )
        env.recorder_manager.export_episodes()
        if pressed:
            print(f"[RESULT] exported to  : {args.dataset_file}")
        else:
            print(f"[RESULT] NOT exported : pressed=False (skipping save)")
    except Exception as e:
        print(f"[RESULT] export FAILED : {e}")
print("=" * 60)

env.close()
