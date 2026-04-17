#!/bin/bash
# Run the Mimic augmentation pipeline on converted IK-Rel demos.
# 1. Convert joint_pos demos → IK-Rel (offline FK)
# 2. annotate_demos (auto subtask signals from env)
# 3. generate_dataset (mimic augment to 100+ trials)
# 4. (optional) train on augmented dataset
set -e

SRC_IKREL=/isaac-sim/output/callbutton_demos_ikrel/merged.hdf5
ANNOTATED=/isaac-sim/output/callbutton_demos_ikrel/annotated.hdf5
GENERATED=/isaac-sim/output/callbutton_demos_ikrel/generated.hdf5

# 1. Convert
echo "=== Convert joint_pos → IK-Rel ==="
/isaac-sim/python.sh /tmp/convert_to_ik_rel.py

# 2. Annotate (auto mode — uses press_done signal)
echo ""
echo "=== Annotate demos ==="
/isaac-sim/python.sh /workspace/robotis_lab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task RobotisLab-CallButton-Right-OMY-IK-Rel-Mimic-v0 \
  --auto \
  --input_file "${SRC_IKREL}" \
  --output_file "${ANNOTATED}" \
  --enable_cameras --headless

# 3. Generate augmented dataset
echo ""
echo "=== Generate augmented dataset ==="
/isaac-sim/python.sh /workspace/robotis_lab/scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cuda --num_envs 10 \
  --task RobotisLab-CallButton-Right-OMY-IK-Rel-Mimic-v0 \
  --generation_num_trials 100 \
  --input_file "${ANNOTATED}" \
  --output_file "${GENERATED}" \
  --enable_cameras --headless

echo ""
echo "=== Pipeline done ==="
echo "Generated dataset: ${GENERATED}"
