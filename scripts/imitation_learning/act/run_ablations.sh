#!/bin/bash
# Run 4 aux_weight ablations sequentially on the robotis_callbutton task.
# Each run: 1500 epochs, dinov2_small backbone, batch 4, kl_weight 10.
# Assumes act-format dataset already exists at the task's dataset_dir
# (see act_repo/constants.py SIM_TASK_CONFIGS["robotis_callbutton"]).

set -e

ACT_REPO=/mnt/Dataset/act_repo
PYTHON=/isaac-sim/python.sh
TASK=robotis_callbutton
EPOCHS=1500
SEED=0
BACKBONE=dinov2_small
KL_WEIGHT=10
CHUNK_SIZE=100
HIDDEN_DIM=512
DIM_FF=3200
BATCH=4
LR=1e-5

CKPT_BASE=/mnt/Dataset/act_ckpts/aux_ablation
mkdir -p "$CKPT_BASE"
chmod 777 "$CKPT_BASE"

for w in 0.0 0.1 0.5 1.0; do
  tag=$(printf "auxw%g" "$w" | tr -d '.')   # auxw0, auxw01, auxw05, auxw1
  ckpt_dir="$CKPT_BASE/$tag"
  log="$CKPT_BASE/$tag.log"
  mkdir -p "$ckpt_dir"
  chmod 777 "$ckpt_dir"
  echo "[ABL] $(date) start aux_weight=$w -> $ckpt_dir"
  cd "$ACT_REPO" && "$PYTHON" imitate_episodes.py \
    --ckpt_dir "$ckpt_dir" \
    --policy_class ACT \
    --task_name "$TASK" \
    --batch_size "$BATCH" \
    --seed "$SEED" \
    --num_epochs "$EPOCHS" \
    --lr "$LR" \
    --kl_weight "$KL_WEIGHT" \
    --chunk_size "$CHUNK_SIZE" \
    --hidden_dim "$HIDDEN_DIM" \
    --dim_feedforward "$DIM_FF" \
    --backbone "$BACKBONE" \
    --aux_weight "$w" > "$log" 2>&1
  echo "[ABL] $(date) done aux_weight=$w"
done
echo "[ABL] all 4 ablations finished"
