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

to_docker_path() {
    local host_path="$1"

    if command -v cygpath >/dev/null 2>&1; then
        cygpath -m "$host_path"
    else
        echo "$host_path"
    fi
}

docker_cli() {
    MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" docker "$@"
}

build_image() {
    docker_cli build -t "$IMAGE_NAME" .
}

create_smoke() {
    echo "Creating smoke dataset..."
    bash scripts/create_smoke.sh
}

train_full() {
    local full_data model_full
    local full_data_docker model_full_docker

    full_data="$(get_absolute_path "$FULL_DATA_REL")"
    model_full="$(get_absolute_path ".")/${MODEL_FULL_REL}"
    full_data_docker="$(to_docker_path "$full_data")"
    model_full_docker="$(to_docker_path "$model_full")"

    ensure_directory "$model_full"

    docker_cli run --rm \
        -v "${full_data_docker}:/challenge/training_data:ro" \
        -v "${model_full_docker}:/challenge/model" \
        "$IMAGE_NAME" \
        python train_model.py -d training_data -m model -v
}

train_smoke() {
    local smoke_data model_smoke
    local smoke_data_docker model_smoke_docker

    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    model_smoke="$(get_absolute_path ".")/${MODEL_SMOKE_REL}"
    smoke_data_docker="$(to_docker_path "$smoke_data")"
    model_smoke_docker="$(to_docker_path "$model_smoke")"

    ensure_directory "$model_smoke"

    docker_cli run --rm \
        -v "${smoke_data_docker}:/challenge/training_data:ro" \
        -v "${model_smoke_docker}:/challenge/model" \
        "$IMAGE_NAME" \
        python train_model.py -d training_data -m model -v
}

run_full() {
    local full_data model_full out_full
    local full_data_docker model_full_docker out_full_docker

    full_data="$(get_absolute_path "$FULL_DATA_REL")"
    model_full="$(get_absolute_path "$MODEL_FULL_REL")"
    out_full="$(get_absolute_path ".")/${OUT_FULL_REL}"
    full_data_docker="$(to_docker_path "$full_data")"
    model_full_docker="$(to_docker_path "$model_full")"
    out_full_docker="$(to_docker_path "$out_full")"

    ensure_directory "$out_full"

    docker_cli run --rm \
        -v "${full_data_docker}:/challenge/holdout_data:ro" \
        -v "${model_full_docker}:/challenge/model:ro" \
        -v "${out_full_docker}:/challenge/holdout_outputs" \
        "$IMAGE_NAME" \
        python run_model.py -d holdout_data -m model -o holdout_outputs -v
}

run_smoke() {
    local smoke_data model_smoke out_smoke
    local smoke_data_docker model_smoke_docker out_smoke_docker

    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    model_smoke="$(get_absolute_path "$MODEL_SMOKE_REL")"
    out_smoke="$(get_absolute_path ".")/${OUT_SMOKE_REL}"
    smoke_data_docker="$(to_docker_path "$smoke_data")"
    model_smoke_docker="$(to_docker_path "$model_smoke")"
    out_smoke_docker="$(to_docker_path "$out_smoke")"

    ensure_directory "$out_smoke"

    docker_cli run --rm \
        -v "${smoke_data_docker}:/challenge/holdout_data:ro" \
        -v "${model_smoke_docker}:/challenge/model:ro" \
        -v "${out_smoke_docker}:/challenge/holdout_outputs" \
        "$IMAGE_NAME" \
        python run_model.py -d holdout_data -m model -o holdout_outputs -v
}

# =====================
# DEVELOPMENT MODE (NO REBUILD)
# =====================

train_dev() {
    local code_path smoke_data model_smoke
    local code_path_docker smoke_data_docker

    code_path="$(get_absolute_path ".")"
    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    model_smoke="${code_path}/${MODEL_SMOKE_REL}"
    code_path_docker="$(to_docker_path "$code_path")"
    smoke_data_docker="$(to_docker_path "$smoke_data")"

    ensure_directory "$model_smoke"

    docker_cli run --rm \
        -v "${code_path_docker}:/challenge" \
        -v "${smoke_data_docker}:/challenge/data_smoke:ro" \
        "$IMAGE_NAME" \
        python train_model.py -d /challenge/data_smoke -m /challenge/model_smoke -v
}

run_dev() {
    local code_path smoke_data out_smoke
    local code_path_docker smoke_data_docker

    code_path="$(get_absolute_path ".")"
    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    out_smoke="${code_path}/${OUT_SMOKE_REL}"
    code_path_docker="$(to_docker_path "$code_path")"
    smoke_data_docker="$(to_docker_path "$smoke_data")"

    ensure_directory "$out_smoke"

    docker_cli run --rm \
        -v "${code_path_docker}:/challenge" \
        -v "${smoke_data_docker}:/challenge/data_smoke:ro" \
        "$IMAGE_NAME" \
        python run_model.py -d /challenge/data_smoke -m /challenge/model_smoke -o /challenge/outputs_smoke -v
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
