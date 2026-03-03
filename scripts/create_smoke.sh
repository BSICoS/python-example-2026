#!/usr/bin/env bash
set -euo pipefail

# ============================================
# Create smoke training dataset
# ============================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# IMPORTANT:
# Each team member can modify this path to
# match their local dataset location.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
FULL_DATA_PATH="${FULL_DATA_PATH:-data/training_set}"  # Override with env var if needed

SMOKE_PATH="data/training_smoke"
N_RECORDS="${N_RECORDS:-5}"

echo "Creating smoke dataset..."
echo "Source: ${FULL_DATA_PATH}"
echo "Destination: ${SMOKE_PATH}"

rm -rf "${SMOKE_PATH}"
mkdir -p "${SMOKE_PATH}"

# Copy demographics
cp "${FULL_DATA_PATH}/demographics.csv" "${SMOKE_PATH}/demographics.csv"

# Select first N EDF files
while IFS= read -r file_path; do
    rel_path="${file_path#${FULL_DATA_PATH}/}"
    target_path="${SMOKE_PATH}/${rel_path}"
    mkdir -p "$(dirname "${target_path}")"
    cp "${file_path}" "${target_path}"
done < <(
    find "${FULL_DATA_PATH}/physiological_data" -type f -name "*.edf" | sort | head -n "${N_RECORDS}"
)

# Copy full annotation folders (simpler and robust)
if [[ -d "${FULL_DATA_PATH}/algorithmic_annotations" ]]; then
    cp -R "${FULL_DATA_PATH}/algorithmic_annotations" "${SMOKE_PATH}/algorithmic_annotations"
fi

if [[ -d "${FULL_DATA_PATH}/human_annotations" ]]; then
    cp -R "${FULL_DATA_PATH}/human_annotations" "${SMOKE_PATH}/human_annotations"
fi

echo "Smoke dataset created successfully."
