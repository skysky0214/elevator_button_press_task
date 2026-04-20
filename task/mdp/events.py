"""Custom events for the elevator call-button task.

randomize_elevator_pose : shift elevator USD in xyz per env at reset. Button
  position follows along (observations read its current xform each step).

reset_robot_at_hall_side_of_button : place the robot in front of the call
  button with standoff / lateral / yaw randomization — mirrors
  compute_robot_pose + write_root_pose_to_sim in omy_f3m_press_diffik.py.

Both run at `reset` mode. `randomize_elevator_pose` MUST be listed before
`reset_robot_at_hall_side_of_button` so the robot placement sees the already-
shifted elevator.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Default elevator pose (matches diffik script + elevator_call_env_cfg).
_ELEVATOR_DEFAULT_POS = (1.8, 0.0, 0.0)


_GRIPPER_JOINT_NAMES = ("rh_r1_joint", "rh_r2", "rh_l1", "rh_l2")
_GRIPPER_CLOSED_RAD = 1.0


def patch_gripper_drives(env: "ManagerBasedRLEnv", env_ids: torch.Tensor):
    """Remove the broken mimic API on OMY gripper joints and install standalone
    DriveAPIs with closed-targets, mirroring patch_gripper() in
    omy_f3m_press_diffik.py.

    After USD import the three "slave" finger joints (rh_r2, rh_l1, rh_l2)
    are declared as PhysxMimicJoint of rh_r1_joint, but the mimic reference
    is broken so neither the mimic nor PD tracking actually closes them. We
    strip the mimic API, force finite revolute limits, and install a force
    drive (stiffness=1500, damping=60, target=closed) on all four joints.
    Runs once at `startup` so it takes effect before physics initializes."""
    import omni.usd
    from pxr import UsdPhysics as UP, UsdGeom  # noqa: F401
    try:
        from pxr import PhysxSchema  # type: ignore
    except Exception:
        PhysxSchema = None  # patch still works — we just skip API removal

    stage = omni.usd.get_context().get_stage()
    limit_deg = float(np.degrees(1.13514578304))  # URDF upper limit
    closed_deg = float(np.degrees(_GRIPPER_CLOSED_RAD))

    # Broad-traverse the stage and match any prim whose name is one of the
    # gripper joints and whose type is a RevoluteJoint. OMY.usd puts joints
    # at different paths depending on how the URDF was imported, so a fixed
    # path pattern isn't reliable.
    targets = set(_GRIPPER_JOINT_NAMES)
    patched_paths = []
    for prim in stage.Traverse():
        if prim.GetName() not in targets:
            continue
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue
        # Must live under some /World/envs/.../Robot subtree.
        path_str = str(prim.GetPath())
        if "/Robot" not in path_str:
            continue

        rev = UP.RevoluteJoint(prim)
        rev.CreateLowerLimitAttr().Set(0.0)
        rev.CreateUpperLimitAttr().Set(limit_deg)

        # Only strip mimic — the ImplicitActuator from OMY_CFG already installs
        # proper DriveAPIs; adding another one here conflicts with it and
        # corrupts PhysX articulation state on GPU.
        if PhysxSchema is not None:
            for schema_name in list(prim.GetAppliedSchemas()):
                if "Mimic" in schema_name:
                    instance = schema_name.split(":", 1)[1] if ":" in schema_name else ""
                    try:
                        prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, instance)
                    except Exception:
                        pass
        for an in list(prim.GetPropertyNames()):
            if "mimic" in an.lower():
                try:
                    prim.RemoveProperty(an)
                except Exception:
                    pass
        patched_paths.append(path_str)
    print(f"[patch_gripper_drives] patched {len(patched_paths)} joint drive(s):")
    for p in patched_paths:
        print(f"    {p}")


def randomize_elevator_pose(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    z_range: tuple[float, float] = (-0.10, 0.10),
):
    """Shift the elevator prim's root translation per env. Used to vary both
    the button height (z) and its planar placement (x, y) so demos cover
    different approach distances and button elevations."""
    import omni.usd
    from pxr import UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()
    ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)

    for env_id in ids:
        elev_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Elevator")
        if not elev_prim.IsValid():
            continue
        xform = UsdGeom.Xformable(elev_prim)
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xform.AddTranslateOp()
        dx = float(np.random.uniform(x_range[0], x_range[1]))
        dy = float(np.random.uniform(y_range[0], y_range[1]))
        dz = float(np.random.uniform(z_range[0], z_range[1]))
        translate_op.Set(Gf.Vec3d(
            _ELEVATOR_DEFAULT_POS[0] + dx,
            _ELEVATOR_DEFAULT_POS[1] + dy,
            _ELEVATOR_DEFAULT_POS[2] + dz,
        ))


def reset_robot_at_hall_side_of_button(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    standoff_range: tuple[float, float] = (0.4, 0.8),
    lateral_range: tuple[float, float] = (-0.25, 0.25),
    yaw_offset_range_deg: tuple[float, float] = (-25.0, 25.0),
    pedestal_height: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Place the robot (and pedestal) in the hall near the call plate, with
    per-env randomization so approach trajectories vary across demos.

    Randomized axes (in the robot-relative "hall" frame):
      - standoff   : distance along the outward plate normal (away from wall)
      - lateral    : offset along the wall-tangent direction
      - yaw_offset : rotation around Z relative to "facing the button exactly"
    """
    import omni.usd
    from pxr import UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()
    robot: Articulation = env.scene[asset_cfg.name]

    ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)

    poses = []
    for env_id in ids:
        btn_prim = stage.GetPrimAtPath(
            f"/World/envs/env_{env_id}/Elevator/Elevator/HallExterior/CallBtn_0"
        )
        plate_prim = stage.GetPrimAtPath(
            f"/World/envs/env_{env_id}/Elevator/Elevator/HallExterior/CallPlate"
        )
        if not btn_prim.IsValid():
            poses.append(None)
            continue

        btn_t = UsdGeom.Xformable(btn_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
        if plate_prim.IsValid():
            plate_t = UsdGeom.Xformable(plate_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
        else:
            plate_t = Gf.Vec3d(btn_t[0], btn_t[1], btn_t[2])

        normal = np.array([btn_t[0] - plate_t[0], btn_t[1] - plate_t[1]], dtype=np.float64)
        n = float(np.linalg.norm(normal))
        if n > 1e-4:
            normal /= n
        else:
            # plate absent/coincident — assume outward is -x (right-side panel in world after yaw)
            normal = np.array([-1.0, 0.0])

        # Sample randomized standoff + lateral + yaw offset per env.
        s = float(np.random.uniform(standoff_range[0], standoff_range[1]))
        lat = float(np.random.uniform(lateral_range[0], lateral_range[1]))
        dyaw = float(np.deg2rad(np.random.uniform(yaw_offset_range_deg[0], yaw_offset_range_deg[1])))

        # Tangent along the wall (perpendicular to outward normal, in xy plane).
        tangent = np.array([-normal[1], normal[0]])

        rx = btn_t[0] + normal[0] * s + tangent[0] * lat
        ry = btn_t[1] + normal[1] * s + tangent[1] * lat
        # Base yaw: face the button exactly. Then add random offset.
        yaw = float(np.arctan2(-normal[1], -normal[0])) + dyaw
        qw = float(np.cos(yaw / 2.0))
        qz = float(np.sin(yaw / 2.0))
        poses.append((rx, ry, pedestal_height, qw, qz))

    # Write robot root poses
    device = env.device
    valid_ids = [i for i, p in zip(ids, poses) if p is not None]
    if not valid_ids:
        return
    rows = []
    for p in poses:
        if p is None:
            continue
        rx, ry, rz, qw, qz = p
        rows.append([rx, ry, rz, qw, 0.0, 0.0, qz])
    pose_tensor = torch.tensor(rows, device=device, dtype=torch.float32)
    robot.write_root_pose_to_sim(pose_tensor, env_ids=torch.tensor(valid_ids, device=device))

    # Move the pedestal (kinematic rigid) via physics-state write so it
    # visually sticks under the dynamically-placed robot.
    try:
        pedestal = env.scene["pedestal"]
    except KeyError:
        pedestal = None
    if pedestal is not None:
        ped_rows = []
        ped_env_ids = []
        for env_id, p in zip(ids, poses):
            if p is None:
                continue
            rx, ry, _, _, _ = p
            ped_rows.append([rx, ry, pedestal_height * 0.5, 1.0, 0.0, 0.0, 0.0])
            ped_env_ids.append(env_id)
        if ped_rows:
            ped_pose_tensor = torch.tensor(ped_rows, device=device, dtype=torch.float32)
            pedestal.write_root_pose_to_sim(
                ped_pose_tensor, env_ids=torch.tensor(ped_env_ids, device=device)
            )


def randomize_init_joints(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    joint1_range_rad: tuple[float, float] = (-1.0, 1.0),
    joint2_range_rad: tuple[float, float] = (-0.5, 0.3),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Perturb initial arm joint_pos at reset to broaden demo coverage.

    joint1 (base yaw) offset up to ~±60° so the arm can start facing any
    direction, and joint2 (shoulder pitch) offset so arm can start folded
    differently. Remaining joints stay at OMY default HOME pose.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    ids_list = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
    if not ids_list:
        return
    device = env.device
    env_ids_t = torch.tensor(ids_list, device=device)

    # Clone current joint_pos / joint_vel for selected envs
    jpos = robot.data.default_joint_pos[env_ids_t].clone()
    jvel = robot.data.default_joint_vel[env_ids_t].clone()

    j1_ids = robot.find_joints(["joint1"])[0]
    j2_ids = robot.find_joints(["joint2"])[0]

    for i, _ in enumerate(ids_list):
        d1 = float(np.random.uniform(joint1_range_rad[0], joint1_range_rad[1]))
        d2 = float(np.random.uniform(joint2_range_rad[0], joint2_range_rad[1]))
        jpos[i, j1_ids[0]] += d1
        jpos[i, j2_ids[0]] += d2

    robot.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids_t)


def cache_button_rest_pos(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
):
    """Cache CallBtn_0 rest world position per env at reset. Runs AFTER
    randomize_elevator_pose so it reflects the shifted elevator. Used by
    call_button_depression observation and success termination."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
    if not hasattr(env, "button_rest_pos_w"):
        env.button_rest_pos_w = torch.zeros((env.num_envs, 3), device=env.device)

    for env_id in ids:
        btn_prim = stage.GetPrimAtPath(
            f"/World/envs/env_{env_id}/Elevator/Elevator/HallExterior/CallBtn_0"
        )
        if not btn_prim.IsValid():
            continue
        t = UsdGeom.Xformable(btn_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
        env.button_rest_pos_w[env_id, 0] = float(t[0])
        env.button_rest_pos_w[env_id, 1] = float(t[1])
        env.button_rest_pos_w[env_id, 2] = float(t[2])
