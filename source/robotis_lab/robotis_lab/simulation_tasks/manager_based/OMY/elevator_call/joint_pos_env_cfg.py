# Joint-position control variant. OMY joints are driven directly.

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from robotis_lab.simulation_tasks.manager_based.OMY.elevator_call import mdp
from robotis_lab.simulation_tasks.manager_based.OMY.elevator_call.elevator_call_env_cfg import (
    FRAME_MARKER_SMALL_CFG,
    ROBOT_XY,
    PEDESTAL_HEIGHT,
    ElevatorCallEnvCfg,
)

from robotis_lab.assets.robots.OMY import OMY_CFG  # isort: skip


@configclass
class OMYElevatorCallEnvCfg(ElevatorCallEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Place OMY on top of the pedestal, matching diffik script. Override
        # the gripper actuator (stiffness/damping/effort) to match diffik's
        # ArticulationCfg — OMY_CFG's stock gripper isn't stiff enough to
        # drive the 4-finger mimic linkage fully closed (tops out around 0.70).
        # Also start with rh_r1_joint = 1.0 (closed) so the press phase runs
        # with a closed fist from step 0, like in omy_f3m_press_diffik.py.
        closed_joint_pos = dict(OMY_CFG.init_state.joint_pos)
        closed_joint_pos["rh_r1_joint"] = 1.0
        closed_joint_pos["rh_r2"] = 1.0
        closed_joint_pos["rh_l1"] = 1.0
        closed_joint_pos["rh_l2"] = 1.0
        self.scene.robot = OMY_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=OMY_CFG.init_state.replace(
                pos=(ROBOT_XY[0], ROBOT_XY[1], PEDESTAL_HEIGHT),
                joint_pos=closed_joint_pos,
            ),
            actuators={
                **OMY_CFG.actuators,
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"],
                    effort_limit_sim=5000.0,
                    velocity_limit_sim=3.0,
                    stiffness=2000000.0,
                    damping=20000.0,
                ),
            },
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
        # Mimic joints on this gripper don't always track the driver (rh_r1_joint)
        # reliably after USD import, so drive all 4 finger joints explicitly.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"],
            open_command_expr={"rh_r1_joint": 1.0, "rh_r2": 1.0, "rh_l1": 1.0, "rh_l2": 1.0},
            close_command_expr={"rh_r1_joint": 1.0, "rh_r2": 1.0, "rh_l1": 1.0, "rh_l2": 1.0},
        )

        # Cameras
        self.scene.cam_wrist = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/OMY/link6/cam_wrist",
            update_period=0.0,
            height=1200, width=1920,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=11.8, focus_distance=200.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, -0.08, 0.07),
                rot=(0.0018, -0.0018, 0.7071, 0.7071),
                convention="isaac",
            ),
        )
        self.scene.cam_top = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/OMY/world/cam_top",
            update_period=0.0,
            height=1200, width=1920,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0, focus_distance=200.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.000233, 0.1499, 0.43953),
                rot=(0.5144, 0.4146, -0.5064, -0.5542),
                convention="isaac",
            ),
        )
        self.scene.cam_belly = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/OMY/world/cam_belly",
            update_period=0.0,
            height=1200, width=1920,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0, focus_distance=200.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.08705, 0.1499, 0.1075),
                rot=(-0.4901, -0.5097, 0.5097, 0.4901),  # tuned in cam_edit_belly.py
                convention="isaac",
            ),
        )

        # EEF frame: mirror the cabinet task so downstream tooling sees the
        # same "end_effector" anchor (link6 with the TCP offset).
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/OMY/world",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/OMY/link6",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, -0.248, 0.0)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/OMY/gripper/rh_p12_rn_l2",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/OMY/gripper/rh_p12_rn_r2",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )
