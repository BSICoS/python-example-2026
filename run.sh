#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <build|train|train-dev|clean>"
    exit 1
fi

COMMAND="$1"

# ============================================
# CONFIGURATION
# ============================================

TRAIN_DATA_REL="D:/data/training_set"

IMAGE_NAME="cinc2026"

MODEL_FULL_REL="model"
FEATURE_CACHE_REL=".feature_cache"

# ============================================
# HELPERS
# ============================================

get_absolute_path() {
    local target_path="$1"
    # Si la ruta ya es absoluta (empieza por C:, D:, X:, etc.) la dejamos tal cual
    if [[ "$target_path" =~ ^[A-Za-z]: ]]; then
        echo "$target_path"
    else
        (cd "$target_path" && pwd)
    fi
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

GPU_ARGS=()

configure_gpu_args() {
    if docker_cli run --rm --gpus all \
        "$IMAGE_NAME" \
        python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
        >/dev/null 2>&1; then
        echo "CUDA GPU detected. Using GPU."
        GPU_ARGS=(--gpus all)
    else
        echo "CUDA GPU not available. Using CPU."
        GPU_ARGS=()
    fi
}

build_image() {
    docker_cli build -t "$IMAGE_NAME" .
}

train_full() {
    local full_data model_full
    local feature_cache
    local full_data_docker model_full_docker feature_cache_docker

    full_data="$(get_absolute_path "$TRAIN_DATA_REL")"
    model_full="$(get_absolute_path ".")/${MODEL_FULL_REL}"
    feature_cache="$(get_absolute_path ".")/${FEATURE_CACHE_REL}"
    full_data_docker="$(to_docker_path "$full_data")"
    model_full_docker="$(to_docker_path "$model_full")"
    feature_cache_docker="$(to_docker_path "$feature_cache")"

    ensure_directory "$model_full"
    ensure_directory "$feature_cache"

    configure_gpu_args

    docker_cli run --rm "${GPU_ARGS[@]}" \
        -v "${full_data_docker}:/challenge/training_data:ro" \
        -v "${model_full_docker}:/challenge/model" \
        -v "${feature_cache_docker}:/challenge/.feature_cache" \
        "$IMAGE_NAME" \
        python train_model.py -d training_data -m model -v
}

# =====================
# DEVELOPMENT MODE (NO REBUILD, FULL DATASETS)
# =====================

train_dev() {
    local code_path full_data model_full
    local code_path_docker full_data_docker

    code_path="$(get_absolute_path ".")"
    full_data="$(get_absolute_path "$TRAIN_DATA_REL")"
    model_full="${code_path}/${MODEL_FULL_REL}"
    code_path_docker="$(to_docker_path "$code_path")"
    full_data_docker="$(to_docker_path "$full_data")"

    ensure_directory "$model_full"

    configure_gpu_args

    docker_cli run --rm "${GPU_ARGS[@]}" \
        -v "${code_path_docker}:/challenge" \
        -v "${full_data_docker}:/challenge/training_data:ro" \
        "$IMAGE_NAME" \
        python train_model.py -d /challenge/training_data -m /challenge/model -v
}

clean_all() {
    rm -rf "$MODEL_FULL_REL"
    echo "Models and outputs removed."
}

case "$COMMAND" in
    build)       build_image ;;
    train)       train_full ;;
    train-dev)   train_dev ;;
    clean)       clean_all ;;
    *)
        echo "Invalid command: $COMMAND"
        echo "Valid commands: build, train, train-dev, clean"
        exit 1
        ;;
esac
