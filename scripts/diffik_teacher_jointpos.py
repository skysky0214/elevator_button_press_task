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
parser.add_argument("--press-depth", type=float, default=0.035)
parser.add_argument("--approach-steps", type=int, default=300)
parser.add_argument("--press-steps", type=int, default=1500)
parser.add_argument("--hold-steps", type=int, default=60)
parser.add_argument("--home-idle-steps", type=int, default=150, help="Hold at HOME after retract")
parser.add_argument("--return-steps", type=int, default=200)
parser.add_argument("--contact-threshold", type=float, default=0.03)
parser.add_argument("--contact-force-threshold", type=float, default=0.5, help="Gripper tip contact force (N) to treat as pressed")
parser.add_argument("--initial-hold-steps", type=int, default=90, help="Hold at home for N steps before approach (3s @ 60Hz)")
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


def _wrap_to_pi(q):
    return (np.asarray(q, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def _solve_ik(target_pos, target_rot, seed_q, n_seeds=20):
    best_q, best_err = None, 1e9
    best_cost = 1e9
    base = _wrap_to_pi(seed_q)
    err_tol = 1e-3
    for s in range(n_seeds):
        if s == 0:
            seed = base.copy()
        elif s % 3 == 1:
            # Every 3rd seed: force joint3 into upper range [π/2, π] to help
            # IKPy find the physically reachable branch (avoids singularity at ~0).
            seed = _wrap_to_pi(base + np.random.uniform(-0.3, 0.3, 6))
            seed[2] = np.random.uniform(np.pi / 2, np.pi)
        else:
            seed = _wrap_to_pi(base + np.random.uniform(-0.3, 0.3, 6))
        init = _make_q(seed)
        ik = _chain.inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_rot,
            orientation_mode="all",
            initial_position=init,
        )
        q_raw = ik[1:7]
        q = _wrap_to_pi(q_raw)
        fk = _chain.forward_kinematics(_make_q(q))
        err = float(np.linalg.norm(fk[:3, 3] - target_pos))
        # Raw (non-wrapped) delta: correctly penalises cross-branch solutions
        # (e.g. joint3 jumping from 2.66 to -2.639 costs 5.3 vs nearby 0.14).
        joint_cost = float(np.linalg.norm(q - base))
        within_limits = np.all(np.abs(q) <= np.pi + 1e-6)
        if within_limits and (err < best_err - 1e-6 or (abs(err - best_err) <= err_tol and joint_cost < best_cost)):
            best_err, best_q, best_cost = err, q, joint_cost
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
# Pull the robot a bit closer to the hall button than the task default.
env_cfg.events.reset_robot_at_hall_side.params["standoff_range"] = (0.55, 0.65)

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
# --- Camera pose logger ---
def _log_cams(tag=""):
    for cn in ["cam_wrist", "cam_top", "cam_belly"]:
        try:
            c = env.scene[cn].data
            p = c.pos_w[0].detach().cpu().numpy()
            q = c.quat_w_world[0].detach().cpu().numpy()  # (w,x,y,z)
            # FOV from cfg
            vw = getattr(env.scene[cn].cfg.spawn, 'focal_length', '?')
            print(f"[CAM] {tag} {cn:9s} pos=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f})  quat(wxyz)=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f})  focal={vw}", flush=True)
        except Exception as e:
            print(f"[CAM] {cn} ERR: {e}", flush=True)
_log_cams("RESET")

robot = env.scene["robot"]
device = env.device
arm_ids = robot.find_joints(ARM_JOINTS)[0]
grip_ids = robot.find_joints(GRIPPER_JOINTS)[0]

# Sync IKPy chain bounds with the simulation's actual joint limits so that
# IK solutions are always physically reachable in the sim.
_arm_limits = robot.data.joint_pos_limits[0, arm_ids].cpu().numpy()  # (6, 2): [lo, hi]
_active_links = [l for l, act in zip(_chain.links, _chain.active_links_mask) if act]
for _j, _link in enumerate(_active_links):
    _link.bounds = (float(_arm_limits[_j, 0]), float(_arm_limits[_j, 1]))
print(f"[SETUP] arm_limits = {np.round(_arm_limits, 3).tolist()}")

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
pre_press_b = np.array([pre_press_x, float(button_base[1]), float(button_base[2]) + 0.11])  # press 2cm higher
press_b = button_base + np.array([args.press_depth, 0.0, 0.11])  # press 11cm higher

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

goal_pre_q, pre_err = _solve_ik(pre_press_b, R_face_button, home_arm, n_seeds=50)
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
print("[PHASE] gripper close")
settle_action = _make_action(home_arm, gripper=1.0)
_env_ids = torch.arange(env.num_envs, device=device, dtype=torch.int64)
_closed_pos = torch.ones(env.num_envs, len(grip_ids), device=device)
_zero_vel   = torch.zeros(env.num_envs, len(grip_ids), device=device)
for i in range(60):
    obs, rew, term, trunc, info = env.step(settle_action)
    # Force all gripper joints to 1.0 directly in the physics sim every step
    robot.write_joint_state_to_sim(
        position=_closed_pos,
        velocity=_zero_vel,
        joint_ids=grip_ids,
        env_ids=_env_ids,
    )
_grip_actual = robot.data.joint_pos[0, grip_ids].detach().cpu().numpy()
print(f"[GRIPPER] closed: actual={np.round(_grip_actual, 3).tolist()}")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Phase 1.5: initial hold — robot stays at home for visual/teleop observability
# -----------------------------------------------------------------------------
if args.initial_hold_steps > 0:
    print(f"[PHASE 1.5: initial hold] {args.initial_hold_steps} steps at home")
    for _ in range(args.initial_hold_steps):
        obs, rew, term, trunc, info = env.step(settle_action)
        _enforce_gripper_closed()


# Shared DLS setup (approach + press phases)
# -----------------------------------------------------------------------------
tip_frame_idx = robot.find_bodies(TIP_LINK)[0][0]
tip_jacobi_idx = tip_frame_idx - 1
PUSH_STEP_M = 0.002
JOINT_CLAMP_RAD = 0.025
DLS_LAMBDA = 0.05

def damped_pinv_torch(J, lam):
    JT = J.transpose(-2, -1)
    eye = torch.eye(J.shape[-2], device=J.device).unsqueeze(0)
    return JT @ torch.inverse(J @ JT + lam * lam * eye)

# Pre-press world position (base-frame → world)
_c, _s = np.cos(robot_yaw), np.sin(robot_yaw)
_R_bw = np.array([[_c, -_s, 0.], [_s, _c, 0.], [0., 0., 1.]])
pre_press_w = (_R_bw @ pre_press_b + robot_pos_w).astype(np.float32)
pre_press_w_t = torch.tensor([pre_press_w], dtype=torch.float32, device=device)
button_w_t = torch.tensor([button_w], dtype=torch.float32, device=device)

# Home tip position: record before approach so retract can return here
home_tip_w = robot.data.body_pos_w[0, tip_frame_idx].detach().cpu().numpy().astype(np.float32)
home_tip_w_t = torch.tensor([home_tip_w], dtype=torch.float32, device=device)
print(f"[SETUP] home_tip_w    = {np.round(home_tip_w, 4).tolist()}")


# -----------------------------------------------------------------------------
# Phase 2: Approach — DLS task-space to pre-press position
# (replaces joint-space IK interpolation to avoid IK branch / singularity issues)
# -----------------------------------------------------------------------------
print(f"[PHASE] approach: up to {args.approach_steps} steps")
current_targets = robot.data.joint_pos[:, arm_ids].clone()
terminated_early = False

for i in range(args.approach_steps):
    tip_w_tensor = robot.data.body_pos_w[:, tip_frame_idx]
    tip_to_pre = pre_press_w_t - tip_w_tensor
    dist_to_pre = float(torch.norm(tip_to_pre[0]))

    if dist_to_pre < 0.015:
        print(f"  [APPROACH DONE] reached pre-press at step {i+1}  dist_to_pre={dist_to_pre:.4f}m")
        break

    push = (tip_to_pre / (torch.norm(tip_to_pre, dim=-1, keepdim=True) + 1e-6)) * PUSH_STEP_M
    J_lin = robot.root_physx_view.get_jacobians()[:, tip_jacobi_idx, :3, arm_ids]
    delta = (damped_pinv_torch(J_lin, DLS_LAMBDA) @ push.unsqueeze(-1)).squeeze(-1)
    delta = torch.clamp(delta, -JOINT_CLAMP_RAD, JOINT_CLAMP_RAD)
    current_targets = current_targets + delta

    obs, rew, term, trunc, info = env.step(_make_action(current_targets[0].cpu().numpy(), gripper=1.0))
    _enforce_gripper_closed()

    if (i + 1) % 20 == 0:
        tip_w = robot.data.body_pos_w[0, tip_frame_idx].detach().cpu().numpy()
        dist = float(np.linalg.norm(tip_w - button_w))
        q_actual = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
        q_str = " ".join(f"{v:6.3f}" for v in q_actual)
        print(f"  approach step={i+1:3d}/{args.approach_steps}  pre_dist={dist_to_pre:.3f}m  tip_dist={dist:.3f}m  q=[{q_str}]")

    if _did_terminate(term, trunc):
        print(f"  early termination during approach at step {i}")
        terminated_early = True
        break


# -----------------------------------------------------------------------------
# Phase 3: Press — DLS micro-thrust toward button
# -----------------------------------------------------------------------------
pressed = False
home_reached = False
min_dist = float("inf")
if not terminated_early:
    print(f"[PHASE] press (micro-thrust): up to {args.press_steps} steps")
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

        if (i + 1) % 20 == 0:
            q_actual = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
            q_str = " ".join(f"{v:6.3f}" for v in q_actual)
            _fmag = 0.0
            try:
                _cs = env.scene["tip_contact"]
                _f = _cs.data.net_forces_w
                if _f is not None:
                    _fmag = float(torch.linalg.norm(_f, dim=-1).max(dim=-1).values[0])
            except (KeyError, AttributeError):
                pass
            print(f"  press step={i+1:4d}  dist={dist:.4f}m  force={_fmag:.3f}N  stall={stall}  q=[{q_str}]")

        # Read gripper-tip contact force
        contact_mag = 0.0
        try:
            cs = env.scene["tip_contact"]
            forces = cs.data.net_forces_w  # (N_env, N_body, 3)
            if forces is not None:
                contact_mag = float(torch.linalg.norm(forces, dim=-1).max(dim=-1).values[0])
        except (KeyError, AttributeError):
            pass

        if contact_mag > args.contact_force_threshold:
            print(f"  [PRESSED] force-contact at step {i+1} force={contact_mag:.3f}N dist={dist:.4f}m")
            pressed = True
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

    print(f"[PHASE] retract: up to {args.return_steps * 2} steps")
    retract_targets = robot.data.joint_pos[:, arm_ids].clone()

    # Step 1: DLS back to pre-press (reverse of press)
    print("  [retract-1] button → pre-press")
    for i in range(args.return_steps):
        tip_w_tensor = robot.data.body_pos_w[:, tip_frame_idx]
        tip_to_pre = pre_press_w_t - tip_w_tensor
        dist_to_pre = float(torch.norm(tip_to_pre[0]))
        if dist_to_pre < 0.015:
            print(f"    pre-press reached at step {i+1}  dist={dist_to_pre:.4f}m")
            break
        push = (tip_to_pre / (torch.norm(tip_to_pre, dim=-1, keepdim=True) + 1e-6)) * PUSH_STEP_M
        J_lin = robot.root_physx_view.get_jacobians()[:, tip_jacobi_idx, :3, arm_ids]
        delta = (damped_pinv_torch(J_lin, DLS_LAMBDA) @ push.unsqueeze(-1)).squeeze(-1)
        delta = torch.clamp(delta, -JOINT_CLAMP_RAD, JOINT_CLAMP_RAD)
        retract_targets = retract_targets + delta
        obs, rew, term, trunc, info = env.step(_make_action(retract_targets[0].cpu().numpy(), gripper=1.0))
        _enforce_gripper_closed()

    # Step 2: DLS back to home tip (reverse of approach)
    print("  [retract-2] pre-press → home")
    for i in range(args.return_steps):
        tip_w_tensor = robot.data.body_pos_w[:, tip_frame_idx]
        tip_to_home = home_tip_w_t - tip_w_tensor
        dist_to_home = float(torch.norm(tip_to_home[0]))
        if dist_to_home < 0.015:
            print(f"    home reached at step {i+1}  dist={dist_to_home:.4f}m")
            home_reached = True
            break
        push = (tip_to_home / (torch.norm(tip_to_home, dim=-1, keepdim=True) + 1e-6)) * PUSH_STEP_M
        J_lin = robot.root_physx_view.get_jacobians()[:, tip_jacobi_idx, :3, arm_ids]
        delta = (damped_pinv_torch(J_lin, DLS_LAMBDA) @ push.unsqueeze(-1)).squeeze(-1)
        delta = torch.clamp(delta, -JOINT_CLAMP_RAD, JOINT_CLAMP_RAD)
        retract_targets = retract_targets + delta
        obs, rew, term, trunc, info = env.step(_make_action(retract_targets[0].cpu().numpy(), gripper=1.0))
        _enforce_gripper_closed()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# (home_reached flag initialized earlier)
# Phase 7: HOME IDLE — stay at home pose so model learns "episode complete" state
# -----------------------------------------------------------------------------
if args.home_idle_steps > 0:
    print(f"[PHASE] home_idle: {args.home_idle_steps} steps")
    for _ in range(args.home_idle_steps):
        obs, rew, term, trunc, info = env.step(_make_action(home_arm, gripper=1.0))


# Summary
# -----------------------------------------------------------------------------
tip_w_final = robot.data.body_pos_w[0, robot.find_bodies(TIP_LINK)[0][0]].detach().cpu().numpy()
final_dist = float(np.linalg.norm(tip_w_final - button_w))

print("")
print("=" * 60)
if 0 <= args.usd_index < len(USD_IDS):
    print(f"[RESULT] USD_{USD_IDS[args.usd_index]:03d}")
# Read FSM state for success conditions
led_lit = False
door_open = False
try:
    _fsm = getattr(env, "_fsm_elevator", None)
    if _fsm is not None:
        # Find target (call) button index for env 0
        _btns = _fsm.button.btn_infos[0] if _fsm.button.btn_infos else []
        _call_idxs = [_i for _i, info in enumerate(_btns) if info["kind"] == "call"]
        if _call_idxs:
            _t = _call_idxs[0]
            led_lit = bool(_fsm.button.lit[0][_t])
        _ds = int(_fsm.door.state[0].item()) if hasattr(_fsm.door.state[0], "item") else int(_fsm.door.state[0])
        door_open = _ds >= 2
except Exception as _e:
    print(f"[RESULT] state read ERR: {_e}")

success = bool(pressed and led_lit and door_open and home_reached)

print(f"[RESULT] pressed      : {pressed}")
print(f"[RESULT] led_lit      : {led_lit}")
print(f"[RESULT] door_open    : {door_open}")
print(f"[RESULT] home_reached : {home_reached}")
print(f"[RESULT] success      : {success}")
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
                ep.success = bool(success)
        env.recorder_manager.cfg.dataset_export_mode = (
            DatasetExportMode.EXPORT_ALL if success else DatasetExportMode.EXPORT_NONE
        )
        env.recorder_manager.export_episodes()
        if success:
            print(f"[RESULT] exported to  : {args.dataset_file}")
        else:
            print(f"[RESULT] NOT exported : success=False (skipping save)")
    except Exception as e:
        print(f"[RESULT] export FAILED : {e}")
print("=" * 60)

env.close()
