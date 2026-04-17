#!/bin/bash
# Collect 10 visuomotor demos (joint_pos + cam_wrist + cam_top) headless.
set -u
OUT=/isaac-sim/output/callbutton_demos_cam
SUMMARY=${OUT}/summary.log
mkdir -p "${OUT}"
: > "${SUMMARY}"

USD_IDS=(1 19 22 35 42 46 81 85 88 92)
for IDX in "${!USD_IDS[@]}"; do
    USD=${USD_IDS[$IDX]}
    FILE="${OUT}/demo_$(printf '%02d' "${IDX}")_usd$(printf '%03d' "${USD}").hdf5"
    LOG="${OUT}/demo_$(printf '%02d' "${IDX}")_usd$(printf '%03d' "${USD}").log"

    echo ""
    echo "=========================================="
    echo " DEMO ${IDX}/10  USD_$(printf '%03d' "${USD}")"
    echo "=========================================="

    /isaac-sim/python.sh /tmp/diffik_teacher_jointpos.py \
        --usd-index "${IDX}" --headless --enable_cameras \
        --dataset-file "${FILE}" > "${LOG}" 2>&1

    RESULT=$(grep -E "^\[RESULT\] pressed" "${LOG}" | head -1)
    MIN=$(grep -E "^\[RESULT\] min_distance" "${LOG}" | head -1)
    SAVED=$([[ -f "${FILE}" ]] && echo "saved" || echo "NOT saved")
    SIZE=$([[ -f "${FILE}" ]] && du -h "${FILE}" | awk '{print $1}' || echo "-")
    echo "  ${RESULT}  |  ${MIN}  |  ${SAVED} (${SIZE})"
    printf "USD_%03d  %-32s  %-30s  %s  %s\n" \
        "${USD}" "${RESULT#[RESULT] }" "${MIN#[RESULT] }" "${SAVED}" "${SIZE}" >> "${SUMMARY}"
done

echo ""
echo "=========================================="
cat "${SUMMARY}"
echo "=========================================="
ls -lh "${OUT}"/*.hdf5 2>/dev/null | awk '{print $5, $NF}'
