# ACT Training Pipeline for Elevator Call-Button

This directory contains ACT (Action Chunking Transformer) specific files used to train and roll out
the v5 model on the `RobotisLab-CallButton-Right-OMY-v0` sim task.

## Files
- `play_act.py` — Isaac Sim rollout script (loads ACT checkpoint, runs temporal-ensemble inference).
- `convert_v5_to_act.py` — Converts per-demo HD robomimic hdf5s (from teacher collection) into ACT's
  per-episode format (`episode_N.hdf5` with `/observations/qpos`, `/observations/images/<cam>`,
  `/action`). Resizes images to 240×320 and fuses `call_button_lit` into qpos (state_dim = 8).
- `imitate_episodes.py` — Patched ACT trainer. Adds `--from_ckpt` / `--start_epoch` for resume,
  strips resume args before DETR's argparse, and uses `_ep_rel` for train_history indexing on resume.
- `detr_vae.py` — DETR VAE model with `state_dim = 8` (7 joint + 1 LED).
- `constants.py` — `SIM_TASK_CONFIGS['robotis_callbutton']` entry registering dataset path,
  episode length, cameras, num_episodes.
- `sim_env.py` — Stub for ACT's ALOHA sim env (we use Isaac Sim for rollout).

## Pipeline overview
1. Collect HD demos with `scripts/diffik_teacher_jointpos.py` (force-contact press,
   `call_button_lit` observation auto-recorded).
2. Convert: `python convert_v5_to_act.py --src_dir <hd_demo_dir> --out <act_data_dir>`.
3. Train: `python imitate_episodes.py --task_name robotis_callbutton --policy_class ACT --chunk_size 50 --batch_size 8 --num_epochs 2000 --lr 1e-5 --seed 0 --ckpt_dir <out>`.
4. Rollout: `python play_act.py --task RobotisLab-CallButton-Right-OMY-v0 --ckpt <ckpt> --stats <pkl> --temporal_agg --enable_cameras`.

## Notes on model layout
- qpos: 7 robot joint positions + 1 LED state (on/off) = 8-dim.
- action: 6 arm + 1 gripper = 7-dim; padded to 8-dim for ACT's shared state_dim.
  The 8th action channel is a zero placeholder; sliced off at rollout before `env.step`.
- Image resolution used for training: 240 × 320, 3 cameras (wrist, top, belly).
- CVAE latent dim 32, KL weight 10, encoder layers 4, decoder layers 7, hidden 512, dim_feedforward 3200.
