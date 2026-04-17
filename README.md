# OMY Elevator Call-Button Press Task

A visuomotor imitation learning task for ROBOTIS OMY 6-DOF arm pressing an
elevator hall call button inside Isaac Sim, built on top of
[ROBOTIS-GIT/robotis_lab](https://github.com/ROBOTIS-GIT/robotis_lab) +
[Isaac Lab](https://github.com/isaac-sim/IsaacLab).

## Overview

- **Task**: OMY arm presses `CallBtn_0` on the outside panel of an elevator.
- **Scene**: ground + dome light + procedurally-varied elevator USD (10 spread-sampled variants) + pedestal + OMY.
- **Randomization (per reset)**:
  - Elevator xyz within ±10 cm
  - Robot standoff 0.4 – 0.8 m from button
  - Robot lateral ±25 cm, yaw ±25°
- **Scripted teacher (diffik)**: IKPy multi-seed + joint-space smoothstep + DLS micro-thrust.
- **Visuomotor recording**: wrist cam + top cam RGB @ 240×320, joint state, actions, initial state, states.
- **Trained policy**: BC-RNN-GMM with ResNet18 visual encoder (via robomimic).

## Repo layout

```
elevator_button_press_task/
├── task/                          # Isaac Lab env registration for the task
│   ├── __init__.py                # gym.register — joint_pos / IK-Rel / IK-Rel-Mimic variants
│   ├── elevator_call_env_cfg.py   # Scene + Obs + Events + Terminations (base)
│   ├── joint_pos_env_cfg.py       # Concrete env with OMY + joint position action + cameras
│   ├── ik_rel_env_cfg.py          # IK-Rel action variant (needed for Mimic)
│   ├── mimic_env.py               # Mimic hooks (EEF pose, subtask signals, object poses)
│   ├── mimic_env_cfg.py           # Mimic config (subtask, datagen)
│   ├── mdp/observations.py        # Custom obs: call_button_pos, EEF pose, joint_pos_target, …
│   ├── mdp/events.py              # Robot dynamic placement at hall side of button, elevator randomization
│   └── agents/robomimic/bc_rnn_image.json  # BC-RNN-GMM visuomotor config
├── scripts/
│   ├── diffik_teacher_jointpos.py # Scripted teacher (diffik-style) — drives joint_pos env
│   ├── diffik_teacher_mimic.py    # Same algorithm, IK-Rel action output (for Mimic)
│   ├── cam_edit_static.py         # GUI helper for tuning cam_wrist / cam_top offset
│   ├── convert_to_ik_rel.py       # Offline joint_pos HDF5 → IK-Rel HDF5 (for Mimic)
│   ├── merge_v2.py                # External-link HDF5 merge (tiny metadata file)
│   ├── collect_10_cam.sh          # Batch: record 10 demos with cameras on 10 different USDs
│   ├── run_mimic_pipeline.sh      # Convert → annotate → generate augmented dataset
│   └── verify_demo.sh             # Dump HDF5 structure + sample frames for inspection
└── docs/
    └── setup.md                   # How to drop the task into a robotis_lab install
```

## Installation

See [docs/setup.md](docs/setup.md).

## Usage

### 1. Record 10 visuomotor demos

```bash
bash scripts/collect_10_cam.sh
# → /isaac-sim/output/callbutton_demos_cam/demo_0X_usdYYY.hdf5
```

### 2. Merge via external-link

```bash
/isaac-sim/python.sh scripts/merge_v2.py
```

### 3. Train BC-RNN (robomimic)

```bash
/isaac-sim/python.sh scripts/imitation_learning/robomimic/train.py \
  --task RobotisLab-CallButton-Right-OMY-v0 \
  --algo bc \
  --dataset /isaac-sim/output/callbutton_demos_cam/merged.hdf5 \
  --log_dir callbutton_bc
```

### 4. (Optional) Mimic augmentation pipeline

```bash
bash scripts/run_mimic_pipeline.sh
```

## Attribution / license

This repository is licensed under **Apache 2.0** (see [LICENSE](LICENSE)).

Derived from the task template structure of
[ROBOTIS-GIT/robotis_lab](https://github.com/ROBOTIS-GIT/robotis_lab)
(Apache 2.0). The OMY robot asset, elevator USDs, and the IL pipeline scripts
(`record_demos.py`, `annotate_demos.py`, `generate_dataset.py`,
`action_data_converter.py`, etc.) are part of `robotis_lab` and are NOT
included in this repo — this repo only contains the new task code and
utilities written on top of them.

The diffik teacher algorithm (`scripts/diffik_teacher_*.py`) is ported from
`isaac_sim_demo/example/omy_f3m_press_diffik.py` (internal tooling).

## Acknowledgments

- **Team lead**: 김혜종 (Hyejong Kim)
- **Co-worker**: 정성훈 (Hun Jung)
- Built during internship at ROBOTIS.
