# Setup

This task is a drop-in extension of
[ROBOTIS-GIT/robotis_lab](https://github.com/ROBOTIS-GIT/robotis_lab) running
inside the Isaac Sim 5.1 / Isaac Lab 2.3 Docker image.

## Prerequisites

- Isaac Sim 5.1 Docker image (with Isaac Lab 2.3) running, with the host
  directory `docker-data/output` bind-mounted to `/isaac-sim/output` inside the
  container.
- A working `robotis_lab` checkout inside the container, e.g. at
  `/workspace/robotis_lab`.
- The OMY asset + `robomimic` + `isaaclab_mimic` already installed (they ship
  with the `robotis_lab` image).

## 1. Drop the task into robotis_lab

Copy the `task/` directory of this repo into the robotis_lab task tree, under
`OMY/elevator_call`:

```bash
TARGET=/workspace/robotis_lab/source/robotis_lab/robotis_lab/simulation_tasks/manager_based/OMY/elevator_call
mkdir -p "${TARGET}"
cp -r task/* "${TARGET}/"
```

The resulting directory should contain:

```
OMY/elevator_call/
├── __init__.py
├── elevator_call_env_cfg.py
├── joint_pos_env_cfg.py
├── ik_rel_env_cfg.py
├── mimic_env.py
├── mimic_env_cfg.py
├── mdp/{observations.py,events.py,__init__.py}
└── agents/robomimic/bc_rnn_image.json
```

After this, the following gym ids become available:

- `RobotisLab-CallButton-Right-OMY-v0` — joint-position control (for BC).
- `RobotisLab-CallButton-Right-OMY-IK-Rel-v0` — IK-Rel control.
- `RobotisLab-CallButton-Right-OMY-IK-Rel-Mimic-v0` — IK-Rel + Mimic hooks.

## 2. Assets

The task expects these to already exist on the host (bind-mounted into
`/isaac-sim/output` or `/workspace`):

- Procedurally-varied elevator USDs referenced by
  `elevator_call_env_cfg.py` (10 variants, path configurable there).
- OMY URDF at `/isaac-sim/output/omy_f3m/omy_f3m_abs.urdf` (used by the
  offline IK-Rel converter — not needed for BC-only pipelines).

These are NOT included in this repository (they belong to the
ROBOTIS / `robotis_lab` asset set).

## 3. Cameras

`joint_pos_env_cfg.py` adds two cameras:

- `cam_wrist` — 240×320 RGB on `link6`, wrist view.
- `cam_top` — 240×320 RGB on the robot base, "head camera" matching the
  physical OMY CAD (Δz=+0.36 m, Δx=−0.15 m from arm base).

Offsets were tuned interactively with `scripts/cam_edit_static.py`. If your
hardware differs, re-tune there and paste the new pose into
`joint_pos_env_cfg.py`.

## 4. Record demos

```bash
bash scripts/collect_10_cam.sh
```

This runs the scripted `diffik` teacher (`diffik_teacher_jointpos.py`) on 10
different elevator USD variants, recording wrist+top RGB, joint state,
actions, initial state, and full states into per-demo HDF5s under
`/isaac-sim/output/callbutton_demos_cam/`.

## 5. Merge demos

```bash
/isaac-sim/python.sh scripts/merge_v2.py
```

External-link merge — the resulting `merged.hdf5` is ~4 KB and points at the
per-demo files.

## 6. Train BC-RNN

```bash
/isaac-sim/python.sh /workspace/robotis_lab/scripts/imitation_learning/robomimic/train.py \
  --task RobotisLab-CallButton-Right-OMY-v0 \
  --algo bc \
  --dataset /isaac-sim/output/callbutton_demos_cam/merged.hdf5 \
  --log_dir callbutton_bc
```

The `robomimic_bc_cfg_entry_point` registered in `task/__init__.py` points at
`agents/robomimic/bc_rnn_image.json` (BC-RNN-GMM + ResNet18 visual encoder,
batch 64, 8 epochs, `num_data_workers=0` to stay within the default 64 MB
`/dev/shm` limit in the Isaac Sim container).

## 7. (Optional) Mimic augmentation

```bash
bash scripts/run_mimic_pipeline.sh
```

This pipeline:

1. Converts joint-position demos to IK-Rel action space offline (IKPy FK),
   adding `eef_pos`/`eef_quat`/`eef_pose` observations.
2. Annotates subtask signals using the env's `press_done` signal.
3. Generates an augmented dataset (100 trials by default, tunable).

Subsequent training uses `RobotisLab-CallButton-Right-OMY-IK-Rel-Mimic-v0`
with the same `bc_rnn_image.json` config.

## Troubleshooting

- **`PermissionError: /isaac-sim/.cache/warp`** — the bind-mounted host dir is
  root-owned. `chmod 777` the host side.
- **DataLoader hangs / `OOM: /dev/shm`** — keep `num_data_workers=0` unless
  you've increased `/dev/shm` (e.g. `--shm-size=8g` on `docker run`).
- **PhysX CUDA illegal memory access at batch ≥ 256** — reduce batch to 64 or
  128. Seen on Isaac Sim 5.1 + torch 2.11 + cu13.
