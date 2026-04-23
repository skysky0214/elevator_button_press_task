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
from isaaclab.sensors import CameraCfg, ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG

from . import mdp

# Interactive elevator (FSM-enabled): buttons light up, doors open, LEDs blink.
import sys
if '/workspace/frontier_simulation/source' not in sys.path:
    sys.path.insert(0, '/workspace/frontier_simulation/source')
from isaac_sim.assets.elevator import ELEVATOR_CFG, ElevatorBehaviorCfg
_SLOW_DOOR_BEHAVIOR = ElevatorBehaviorCfg(
    door_speed=0.35,           # slower open+close (default 0.5)
    auto_close_range=(10.0, 20.0),  # stay open longer
)


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.05, 0.05, 0.05)


# All 100 elevator variants — robot spawns in front of the button regardless of side.
_ELEVATOR_USD_IDS = list(range(1, 101))
import os
# Use the in-container path when available, then fall back to host paths.
_CONTAINER_DIR = "/workspace/robotis_lab/third_party/elevator_setup_new"
_HOST_DIR = "/root/robotis_lab/third_party/elevator_setup_new"
_ALT_DIR = "/home/ub/Downloads/elev_setup_new"
_ELEVATOR_USD_DIR = (
    _CONTAINER_DIR if os.path.isdir(_CONTAINER_DIR) else
    _HOST_DIR if os.path.isdir(_HOST_DIR) else
    _ALT_DIR
)
ELEVATOR_USD_PATHS = [
    f"{_ELEVATOR_USD_DIR}/elevator_setup_{i:03d}.usd" for i in _ELEVATOR_USD_IDS
]

# Match diffik script's elevator pose. 90° yaw so the hall-plate faces -x in world.
ELEVATOR_POS = (1.8, 0.0, 0.0)
ELEVATOR_ROT = (0.7071, 0.0, 0.0, 0.7071)

PEDESTAL_HEIGHT = 0.908  # real robot height
ROBOT_XY = (1.0, 0.5)


@configclass
class ElevatorCallSceneCfg(InteractiveSceneCfg):
    # Populated by agent-level env cfg (joint_pos / ik_rel).
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    # Visuomotor cameras (filled in by agent-level cfg).
    cam_wrist: CameraCfg = MISSING
    cam_top: CameraCfg = MISSING
    cam_belly: CameraCfg = MISSING
    tip_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/OMY/gripper/rh_p12_rn_r2",
        update_period=0.0,
        track_pose=True,
        debug_vis=False,
    )

    # Elevator as a static asset. Buttons have prismatic joints internally; we
    # detect "pressed" via EEF-to-button distance rather than articulation state
    # (the USDs do not declare an ArticulationRoot).
    elevator = ELEVATOR_CFG.replace(
        behavior=_SLOW_DOOR_BEHAVIOR,
        prim_path="{ENV_REGEX_NS}/Elevator",
        init_state=ELEVATOR_CFG.init_state.replace(pos=ELEVATOR_POS, rot=ELEVATOR_ROT),
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
        eef_pos = ObsTerm(func=mdp.ee_pos)
        eef_quat = ObsTerm(func=mdp.ee_quat)
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
        call_button_lit = ObsTerm(func=mdp.call_button_lit)
        cam_belly = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_belly"), "data_type": "rgb", "normalize": False},
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

    init_elevator = EventTerm(func=mdp.init_elevator_fsm, mode="startup")

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Auto-trigger FSM button press when tip is close (interactive elevator only)
    auto_press_button = EventTerm(
        func=mdp.auto_press_button_on_contact,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"distance_threshold": 0.15},
    )


    # Order matters: elevator must shift FIRST so robot placement sees the
    # already-moved button. Isaac Lab runs events in definition order.
    randomize_elevator_pose = EventTerm(
        func=mdp.randomize_elevator_pose,
        mode="reset",
        params={
            "x_range": (0.0, 0.0),
            "y_range": (0.0, 0.0),
            "z_range": (0.0, 0.0),  # elevator pose fixed
        },
    )

    # Dynamically place robot in front of the call button at each reset —
    # mirrors compute_robot_pose + write_root_pose_to_sim in the diffik script.
    reset_robot_at_hall_side = EventTerm(
        func=mdp.reset_robot_at_hall_side_of_button,
        mode="reset",
        params={
            "standoff_range": (0.55, 0.65),
            "lateral_range": (-0.05, 0.05),
            "yaw_offset_range_deg": (0.0, 0.0),
            "pedestal_height": PEDESTAL_HEIGHT,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Success when EEF is within 5cm of CallBtn_0. 5 cm is large enough to
    # detect a surface-level touch without driving the tip into the button mesh.
    success = DoneTerm(
        func=mdp.call_button_pressed,
        params={"distance_threshold": 0.038},
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
