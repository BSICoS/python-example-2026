#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <build|smoke|train|train-smoke|run|run-smoke|train-dev|run-dev|clean>"
    exit 1
fi

COMMAND="$1"

# ============================================
# CONFIGURATION
# ============================================

FULL_DATA_REL="data/training_set"
SMOKE_DATA_REL="data/training_smoke"

IMAGE_NAME="cinc2026"

MODEL_FULL_REL="model"
MODEL_SMOKE_REL="model_smoke"

OUT_FULL_REL="outputs"
OUT_SMOKE_REL="outputs_smoke"

# ============================================
# HELPERS
# ============================================

get_absolute_path() {
    local rel_path="$1"
    (cd "$rel_path" && pwd)
}

ensure_directory() {
    local dir_path="$1"
    mkdir -p "$dir_path"
}

build_image() {
    docker build -t "$IMAGE_NAME" .
}

create_smoke() {
    echo "Creating smoke dataset..."
    bash scripts/create_smoke.sh
}

train_full() {
    local full_data model_full

    full_data="$(get_absolute_path "$FULL_DATA_REL")"
    model_full="$(get_absolute_path ".")/${MODEL_FULL_REL}"

    ensure_directory "$model_full"

    docker run --rm \
        -v "${full_data}:/challenge/training_data:ro" \
        -v "${model_full}:/challenge/model" \
        "$IMAGE_NAME" \
        python train_model.py -d training_data -m model -v
}

train_smoke() {
    local smoke_data model_smoke

    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    model_smoke="$(get_absolute_path ".")/${MODEL_SMOKE_REL}"

    ensure_directory "$model_smoke"

    docker run --rm \
        -v "${smoke_data}:/challenge/training_data:ro" \
        -v "${model_smoke}:/challenge/model" \
        "$IMAGE_NAME" \
        python train_model.py -d training_data -m model -v
}

run_full() {
    local full_data model_full out_full

    full_data="$(get_absolute_path "$FULL_DATA_REL")"
    model_full="$(get_absolute_path "$MODEL_FULL_REL")"
    out_full="$(get_absolute_path ".")/${OUT_FULL_REL}"

    ensure_directory "$out_full"

    docker run --rm \
        -v "${full_data}:/challenge/holdout_data:ro" \
        -v "${model_full}:/challenge/model:ro" \
        -v "${out_full}:/challenge/holdout_outputs" \
        "$IMAGE_NAME" \
        python run_model.py -d holdout_data -m model -o holdout_outputs -v
}

run_smoke() {
    local smoke_data model_smoke out_smoke

    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    model_smoke="$(get_absolute_path "$MODEL_SMOKE_REL")"
    out_smoke="$(get_absolute_path ".")/${OUT_SMOKE_REL}"

    ensure_directory "$out_smoke"

    docker run --rm \
        -v "${smoke_data}:/challenge/holdout_data:ro" \
        -v "${model_smoke}:/challenge/model:ro" \
        -v "${out_smoke}:/challenge/holdout_outputs" \
        "$IMAGE_NAME" \
        python run_model.py -d holdout_data -m model -o holdout_outputs -v
}

# =====================
# DEVELOPMENT MODE (NO REBUILD)
# =====================

train_dev() {
    local code_path smoke_data model_smoke

    code_path="$(get_absolute_path ".")"
    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    model_smoke="${code_path}/${MODEL_SMOKE_REL}"

    ensure_directory "$model_smoke"

    docker run --rm \
        -v "${code_path}:/challenge" \
        -v "${smoke_data}:/challenge/training_data:ro" \
        -v "${model_smoke}:/challenge/model" \
        "$IMAGE_NAME" \
        python train_model.py -d training_data -m model -v
}

run_dev() {
    local code_path smoke_data model_smoke out_smoke

    code_path="$(get_absolute_path ".")"
    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    model_smoke="$(get_absolute_path "$MODEL_SMOKE_REL")"
    out_smoke="${code_path}/${OUT_SMOKE_REL}"

    ensure_directory "$out_smoke"

    docker run --rm \
        -v "${code_path}:/challenge" \
        -v "${smoke_data}:/challenge/holdout_data:ro" \
        -v "${model_smoke}:/challenge/model:ro" \
        -v "${out_smoke}:/challenge/holdout_outputs" \
        "$IMAGE_NAME" \
        python run_model.py -d holdout_data -m model -o holdout_outputs -v
}

clean_all() {
    rm -rf "$MODEL_FULL_REL" "$MODEL_SMOKE_REL" "$OUT_FULL_REL" "$OUT_SMOKE_REL"
    echo "Models and outputs removed."
}

case "$COMMAND" in
    build)       build_image ;;
    smoke)       create_smoke ;;
    train)       train_full ;;
    train-smoke) train_smoke ;;
    run)         run_full ;;
    run-smoke)   run_smoke ;;
    train-dev)   train_dev ;;
    run-dev)     run_dev ;;
    clean)       clean_all ;;
    *)
        echo "Invalid command: $COMMAND"
        echo "Valid commands: build, smoke, train, train-smoke, run, run-smoke, train-dev, run-dev, clean"
        exit 1
        ;;
esac
