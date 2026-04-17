# Copyright 2026 Hwang Yeeun
# SPDX-License-Identifier: Apache-2.0

# OMY elevator hall call-button press environment (base config).
# Scene layout mirrors isaac_sim_demo/example/omy_f3m_press_diffik.py:
#   ground + dome light + elevator USD (MultiAsset over 10 right-side variants)
#   + pedestal cube under the robot + OMY.

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG

from . import mdp


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.05, 0.05, 0.05)


# 10 right-side elevator USDs selected by farthest-point sampling for geometric spread.
_ELEVATOR_USD_IDS = [1, 19, 22, 35, 42, 46, 81, 85, 88, 92]
import os
# Use the in-container path when available (container sees /workspace/),
# otherwise fall back to host path.
_HOST_DIR = "/root/robotis_lab/third_party/elevator_setup"
_CONTAINER_DIR = "/workspace/robotis_lab/third_party/elevator_setup"
_ELEVATOR_USD_DIR = _CONTAINER_DIR if os.path.isdir(_CONTAINER_DIR) else _HOST_DIR
ELEVATOR_USD_PATHS = [
    f"{_ELEVATOR_USD_DIR}/elevator_setup_{i:03d}.usd" for i in _ELEVATOR_USD_IDS
]

# Match diffik script's elevator pose. 90° yaw so the hall-plate faces -x in world.
ELEVATOR_POS = (1.8, 0.0, 0.0)
ELEVATOR_ROT = (0.7071, 0.0, 0.0, 0.7071)

PEDESTAL_HEIGHT = 0.7
ROBOT_XY = (1.0, 0.5)


@configclass
class ElevatorCallSceneCfg(InteractiveSceneCfg):
    # Populated by agent-level env cfg (joint_pos / ik_rel).
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    # Visuomotor cameras (filled in by agent-level cfg).
    cam_wrist: CameraCfg = MISSING
    cam_top: CameraCfg = MISSING

    # Elevator as a static asset. Buttons have prismatic joints internally; we
    # detect "pressed" via EEF-to-button distance rather than articulation state
    # (the USDs do not declare an ArticulationRoot).
    elevator = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Elevator",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[sim_utils.UsdFileCfg(usd_path=p) for p in ELEVATOR_USD_PATHS],
            random_choice=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=ELEVATOR_POS, rot=ELEVATOR_ROT),
    )

    # CallBtn_0 pose read directly from USD xform at each step (see mdp/observations.py).
    # FrameTransformer is not usable here because the button PrismaticJoints in
    # these procedurally-generated USDs use body0=[] (world attachment) with
    # mismatched local anchors, causing PhysX to "snap" the button rigid bodies
    # to the elevator origin and polluting the frame readings.

    # Pedestal the robot stands on. Kinematic rigid so we can move it to
    # follow the dynamically-placed robot via write_root_pose_to_sim.
    pedestal = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/pedestal",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, PEDESTAL_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.28, 0.32)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(ROBOT_XY[0], ROBOT_XY[1], PEDESTAL_HEIGHT * 0.5)),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # State obs (flat vectors).
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        call_button_pos = ObsTerm(func=mdp.call_button_pos)
        rel_ee_call_button_distance = ObsTerm(func=mdp.rel_ee_call_button_distance)
        actions = ObsTerm(func=mdp.last_action)
        # Camera obs (RGB). Only populated when --enable_cameras is used and
        # cam_wrist/cam_top are set in the scene. When enable_cameras is not
        # set, these ObsTerms still exist but return zero-sized tensors.
        cam_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_wrist"), "data_type": "rgb", "normalize": False},
        )
        cam_top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_top"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            # With image terms mixed in, we can no longer concatenate to a flat
            # vector — keep terms as a dict so downstream code (mimic /
            # visuomotor IL) can access cam_wrist / cam_top by name.
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # NOTE: patch_gripper_drives was disabled because stripping the mimic API
    # (even without adding a new DriveAPI) appears to freeze the arm joints'
    # tracking on this Isaac Lab build. We run the demos with the gripper
    # open — good enough for IL data collection.
    #
    # patch_gripper = EventTerm(func=mdp.patch_gripper_drives, mode="startup")

    # Randomize robot friction for robustness; keep elevator friction fixed so
    # prismatic press joint behaves consistently across variants.
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.5, 2.0),
            "dynamic_friction_range": (1.5, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Order matters: elevator must shift FIRST so robot placement sees the
    # already-moved button. Isaac Lab runs events in definition order.
    randomize_elevator_pose = EventTerm(
        func=mdp.randomize_elevator_pose,
        mode="reset",
        params={
            "x_range": (-0.10, 0.10),
            "y_range": (-0.10, 0.10),
            "z_range": (-0.10, 0.10),  # button height variation
        },
    )

    # Dynamically place robot in front of the call button at each reset —
    # mirrors compute_robot_pose + write_root_pose_to_sim in the diffik script.
    reset_robot_at_hall_side = EventTerm(
        func=mdp.reset_robot_at_hall_side_of_button,
        mode="reset",
        params={
            "standoff_range": (0.4, 0.8),
            "lateral_range": (-0.25, 0.25),
            "yaw_offset_range_deg": (-25.0, 25.0),
            "pedestal_height": PEDESTAL_HEIGHT,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Success when EEF is within 3cm of CallBtn_0. Shared across task variants
    # so record_demos.py can detect success and export demos.
    success = DoneTerm(
        func=mdp.call_button_pressed,
        params={"distance_threshold": 0.03},
    )


@configclass
class RewardsCfg:
    """Placeholder. IL pipeline does not use rewards, but ManagerBasedRLEnvCfg
    requires this field to be present."""

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class ElevatorCallEnvCfg(ManagerBasedRLEnvCfg):
    scene: ElevatorCallSceneCfg = ElevatorCallSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 12.0
        self.viewer.eye = (-1.5, 2.0, 2.0)
        self.viewer.lookat = (1.5, 0.0, 1.2)
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
