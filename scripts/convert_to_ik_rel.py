# Copyright 2026 Hwang Yeeun
# SPDX-License-Identifier: Apache-2.0

"""Offline converter: joint_pos HDF5 → IK-Rel HDF5 with eef_pose obs.

For each demo:
  1. Read absolute arm joint positions per step (obs/joint_pos is relative, add default)
  2. IKPy FK → eef_pose in robot base frame
  3. Transform to world frame using robot root pose from initial_state
  4. Compute IK-Rel action = (next_eef_pose - current_eef_pose) / IK_ACTION_SCALE
  5. Write new HDF5 preserving initial_state + states + obs (adding eef_pos/eef_quat)

Output action format matches OMYElevatorCallMimicEnv IK-Rel action space:
  action = [dx, dy, dz, drx, dry, drz, gripper]  (7-dim)
"""
import glob
import os
import shutil
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from ikpy.chain import Chain

URDF_PATH = "/isaac-sim/output/omy_f3m/omy_f3m_abs.urdf"
IKPY_URDF_PATH = "/tmp/omy_arm_only_convert.urdf"
SRC_DIR = "/isaac-sim/output/callbutton_demos_cam"
DST_DIR = "/isaac-sim/output/callbutton_demos_ikrel"

IK_ACTION_SCALE = 0.1
DEFAULT_ARM = np.array([0.0, -1.55, 2.66, -1.1, 1.6, 0.0], dtype=np.float64)


def _build_arm_urdf(src, dst):
    tree = ET.parse(src); root = tree.getroot()
    keep_links = {"world", "link0", "link1", "link2", "link3", "link4", "link5", "link6",
                  "end_effector_flange_link"}
    keep_joints = {"world_fixed", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
                   "end_effector_flange_joint"}
    for e in list(root):
        if e.tag == "link" and e.get("name") not in keep_links:
            root.remove(e)
        elif e.tag == "joint" and e.get("name") not in keep_joints:
            root.remove(e)
    tree.write(dst)


_build_arm_urdf(URDF_PATH, IKPY_URDF_PATH)
_chain = Chain.from_urdf_file(
    IKPY_URDF_PATH, base_elements=["link0"],
    active_links_mask=[False, True, True, True, True, True, True, False],
)


def _make_q(arm6):
    q = np.zeros(8, dtype=np.float64); q[1:7] = arm6; return q


def fk(arm6):
    return _chain.forward_kinematics(_make_q(arm6))  # 4x4 base frame


# ---------- Math helpers ----------
def quat_mul(q1, q2):
    """Multiply two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S])
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S])
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        return np.array([(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S])


def rot_to_axis_angle(R):
    cos_th = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(cos_th)
    if abs(th) < 1e-7:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(th))
    return axis * th


# ---------- Converter ----------
def convert_demo(src_file, dst_file):
    with h5py.File(src_file, "r") as src, h5py.File(dst_file, "w") as dst:
        # Propagate top-level attrs
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        data_grp = dst.create_group("data")
        for k, v in src["data"].attrs.items():
            data_grp.attrs[k] = v

        demo_key = list(src["data"].keys())[0]
        src_demo = src[f"data/{demo_key}"]

        # Read joint sequence (obs/joint_pos is relative to default)
        joint_pos_rel = np.array(src_demo["obs"]["joint_pos"])  # (T, 10)
        T = joint_pos_rel.shape[0]
        arm_rel = joint_pos_rel[:, :6]  # first 6 are arm
        arm_abs = arm_rel + DEFAULT_ARM[None, :]  # (T, 6)

        # Robot root pose from initial_state (assumed constant during episode)
        root_pose = np.array(src_demo["initial_state"]["articulation"]["robot"]["root_pose"])[0]
        # (7,) = [x, y, z, qw, qx, qy, qz]
        root_pos = root_pose[:3]
        root_quat = root_pose[3:7]
        root_R = quat_to_rot(root_quat)

        # FK per step → eef pose (world frame)
        eef_pos_w = np.zeros((T, 3), dtype=np.float32)
        eef_quat_w = np.zeros((T, 4), dtype=np.float32)
        eef_R_world = []
        for t in range(T):
            M_b = fk(arm_abs[t])
            pos_b = M_b[:3, 3]
            R_b = M_b[:3, :3]
            # world: root_pose ∘ base_pose
            pos_w = root_pos + root_R @ pos_b
            R_w = root_R @ R_b
            eef_pos_w[t] = pos_w.astype(np.float32)
            eef_quat_w[t] = rot_to_quat(R_w).astype(np.float32)
            eef_R_world.append(R_w)

        # Compute IK-Rel action per step = (next_eef - current_eef) / scale
        orig_actions = np.array(src_demo["actions"])  # (T, 7) joint_delta + gripper
        gripper = orig_actions[:, -1:]  # (T, 1)

        ik_rel_actions = np.zeros((T, 7), dtype=np.float32)
        for t in range(T):
            if t < T - 1:
                delta_pos = eef_pos_w[t + 1] - eef_pos_w[t]
                delta_R = eef_R_world[t + 1] @ eef_R_world[t].T
                delta_aa = rot_to_axis_angle(delta_R)
            else:
                delta_pos = np.zeros(3)
                delta_aa = np.zeros(3)
            ik_rel_actions[t, :3] = delta_pos / IK_ACTION_SCALE
            ik_rel_actions[t, 3:6] = delta_aa / IK_ACTION_SCALE
            ik_rel_actions[t, 6] = gripper[t, 0]
        ik_rel_actions = np.clip(ik_rel_actions, -1.0, 1.0)

        # Write new demo group
        dst_demo = data_grp.create_group("demo_0")
        for k, v in src_demo.attrs.items():
            dst_demo.attrs[k] = v

        dst_demo.create_dataset("actions", data=ik_rel_actions)
        # keep processed_actions same as orig (for reference)
        if "processed_actions" in src_demo:
            src.copy(f"data/{demo_key}/processed_actions", dst_demo, "processed_actions")
        # copy initial_state + states
        src.copy(f"data/{demo_key}/initial_state", dst_demo, "initial_state")
        if "states" in src_demo:
            src.copy(f"data/{demo_key}/states", dst_demo, "states")

        # Copy existing obs + add eef_pos, eef_quat, eef_pose (7=pos+quat)
        dst_obs = dst_demo.create_group("obs")
        for k in src_demo["obs"]:
            src.copy(f"data/{demo_key}/obs/{k}", dst_obs, k)
        dst_obs.create_dataset("eef_pos", data=eef_pos_w)
        dst_obs.create_dataset("eef_quat", data=eef_quat_w)
        dst_obs.create_dataset("eef_pose", data=np.concatenate([eef_pos_w, eef_quat_w], axis=1))
        # joint_pos_target: use the original joint_pos (absolute) + gripper absolute
        # our env uses 7-dim [joint1..6, rh_r1_joint]
        joint_pos_target = np.zeros((T, 7), dtype=np.float32)
        joint_pos_target[:, :6] = arm_abs.astype(np.float32)
        # rh_r1_joint is index 6 in full obs
        joint_pos_target[:, 6] = joint_pos_rel[:, 6]  # rh_r1_joint relative; close enough for now
        dst_obs.create_dataset("joint_pos_target", data=joint_pos_target)


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    files = sorted(glob.glob(f"{SRC_DIR}/demo_*_usd*.hdf5"))
    if not files:
        raise SystemExit(f"No source demos in {SRC_DIR}")
    for f in files:
        out = f"{DST_DIR}/{os.path.basename(f)}"
        print(f"→ {out}")
        convert_demo(f, out)

    # Merge via external link
    merged = f"{DST_DIR}/merged.hdf5"
    if os.path.exists(merged):
        os.remove(merged)
    with h5py.File(merged, "w") as dst:
        dg = dst.create_group("data")
        with h5py.File(files[0], "r") as src0:
            for k, v in src0["data"].attrs.items():
                dg.attrs[k] = v
        for i, f in enumerate(files):
            dst[f"data/demo_{i}"] = h5py.ExternalLink(
                f"{DST_DIR}/{os.path.basename(f)}", "data/demo_0"
            )

    print(f"\nDone. merged → {merged}")
    with h5py.File(merged, "r") as f:
        d0 = f["data/demo_0"]
        print(f"demo_0 obs keys: {list(d0['obs'].keys())}")
        print(f"demo_0 actions shape: {d0['actions'].shape}")


if __name__ == "__main__":
    main()
