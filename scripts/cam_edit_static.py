# Copyright 2026 Hwang Yeeun
# SPDX-License-Identifier: Apache-2.0

"""Static scene loader for camera tuning.

Spawns ground + light + elevator (single USD) + pedestal + OMY + cam_top only.
No env, no teacher, no events/randomization. User moves cam_top in the GUI
and the script prints its local pose every 2s.

Usage:
    /isaac-sim/python.sh /tmp/cam_edit_static.py --livestream 2
"""
import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd-index", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.enable_cameras = True
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

# Scene constants — must match elevator_call_env_cfg.py.
ELEVATOR_POS = (1.8, 0.0, 0.0)
ELEVATOR_ROT = (0.7071, 0.0, 0.0, 0.7071)
PEDESTAL_HEIGHT = 0.7
# For USD_001: CallBtn_0 ≈ (0.814, 0.822, 1.13) in world.
# Button faces -x, so hall side is at x < 0.814. Place robot 0.5m standoff.
# Robot default yaw = 0 (facing +x = toward button).
ROBOT_XY = (0.3, 0.82)
USD_IDS = [1, 19, 22, 35, 42, 46, 81, 85, 88, 92]
USD_DIR = "/workspace/robotis_lab/third_party/elevator_setup"

from robotis_lab.assets.robots.OMY import OMY_OFF_SELF_COLLISION_CFG as OMY_CFG  # isort: skip


@configclass
class StaticSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))
    elevator = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Elevator",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{USD_DIR}/elevator_setup_{USD_IDS[args.usd_index]:03d}.usd"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=ELEVATOR_POS, rot=ELEVATOR_ROT),
    )
    pedestal = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/pedestal",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, PEDESTAL_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.28, 0.32)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(ROBOT_XY[0], ROBOT_XY[1], PEDESTAL_HEIGHT * 0.5)
        ),
    )
    robot = OMY_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=OMY_CFG.init_state.replace(
            pos=(ROBOT_XY[0], ROBOT_XY[1], PEDESTAL_HEIGHT)
        ),
    )
    cam_top = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/OMY/world/cam_top",
        update_period=0.0,
        height=240, width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, focus_distance=200.0,
            horizontal_aperture=20.955, clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.15, 0.0, 0.36),
            rot=(0.5144, 0.4146, -0.5064, -0.5542),
            convention="isaac",
        ),
    )
    # Wrist camera on link6 (end effector). Initial values copied from
    # pick_place's cam_wrist. User can adjust in GUI and observe the printed
    # local pose below.
    cam_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/OMY/link6/cam_wrist",
        update_period=0.0,
        height=240, width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=11.8, focus_distance=200.0,
            horizontal_aperture=20.955, clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.08, 0.07),
            rot=(0.5, -0.5, -0.5, -0.5),
            convention="isaac",
        ),
    )


sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
sim = sim_utils.SimulationContext(sim_cfg)
scene = InteractiveScene(StaticSceneCfg(num_envs=1, env_spacing=3.0))
sim.reset()

import omni.usd
from pxr import UsdGeom
stage = omni.usd.get_context().get_stage()
CAM_PATHS = {
    "top":   "/World/envs/env_0/Robot/OMY/world/cam_top",
    "wrist": "/World/envs/env_0/Robot/OMY/link6/cam_wrist",
}
cam_prims = {k: stage.GetPrimAtPath(v) for k, v in CAM_PATHS.items()}

print("\n" + "=" * 70)
for k, p in CAM_PATHS.items():
    print(f"[CAM EDIT] {k:5s}: {p}")
print("           Stage tree → select either cam → drag gizmo, or edit")
print("           xformOp:translate / xformOp:orient in Property panel.")
print("           Local poses print every 2s.")
print("=" * 70 + "\n")


def read_local_pose(prim):
    xform = UsdGeom.Xformable(prim)
    pos = (0.0, 0.0, 0.0)
    quat_wxyz = (1.0, 0.0, 0.0, 0.0)
    for op in xform.GetOrderedXformOps():
        n = op.GetName()
        v = op.Get()
        if v is None:
            continue
        if "translate" in n:
            pos = (float(v[0]), float(v[1]), float(v[2]))
        elif "orient" in n:
            quat_wxyz = (float(v.GetReal()),
                         float(v.GetImaginary()[0]),
                         float(v.GetImaginary()[1]),
                         float(v.GetImaginary()[2]))
    return pos, quat_wxyz


last = 0.0
while app.is_running():
    sim.render()
    now = time.time()
    if now - last > 2.0:
        for k, prim in cam_prims.items():
            if not prim.IsValid():
                continue
            pos, quat = read_local_pose(prim)
            print(f"[{k:5s}] pos=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})"
                  f"  quat_wxyz=({quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f})")
        print()
        last = now
    time.sleep(1.0 / 60.0)
