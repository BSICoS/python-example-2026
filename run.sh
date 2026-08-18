#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <build|smoke|train|train-smoke|run|run-smoke|eval|eval-smoke|train-dev|run-dev|eval-dev|clean>"
    exit 1
fi

COMMAND="$1"

# ============================================
# CONFIGURATION
# ============================================

TRAIN_DATA_REL="D:/data/training_set"
RUN_DATA_REL="D:/data/test_set"
SMOKE_DATA_REL="D:/data/training_smoke"

IMAGE_NAME="cinc2026"

MODEL_FULL_REL="model"
MODEL_SMOKE_REL="model_smoke"
FEATURE_CACHE_REL=".feature_cache"

OUT_FULL_REL="outputs"
OUT_SMOKE_REL="outputs_smoke"
DEMOGRAPHICS_FILE="demographics.csv"
PREVALENCE_FILE="prevalence.csv"

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

evaluate_predictions() {
    local code_path="$1"
    local data_path="$2"
    local output_path="$3"
    local label="$4"
    local code_path_docker data_path_docker output_path_docker

    code_path_docker="$(to_docker_path "$code_path")"
    data_path_docker="$(to_docker_path "$data_path")"
    output_path_docker="$(to_docker_path "$output_path")"

    echo "Evaluating ${label} predictions..."
    docker_cli run --rm \
        -v "${code_path_docker}:/challenge" \
        -v "${data_path_docker}:/challenge/eval_data:ro" \
        -v "${output_path_docker}:/challenge/predictions:ro" \
        "$IMAGE_NAME" \
        python evaluate_model.py \
            -d "/challenge/eval_data/${DEMOGRAPHICS_FILE}" \
            -p "/challenge/eval_data/${DEMOGRAPHICS_FILE}" \
            -o "/challenge/predictions/${DEMOGRAPHICS_FILE}"
}

evaluate_predictions_dev() {
    local code_path="$1"
    local data_path="$2"
    local output_path="$3"
    local prevalence_path="$4"
    local label="$5"
    local code_path_docker data_path_docker prevalence_path_docker

    code_path_docker="$(to_docker_path "$code_path")"
    data_path_docker="$(to_docker_path "$data_path")"
    prevalence_path_docker="$(to_docker_path "$prevalence_path")"

    echo "Evaluating ${label} predictions..."
    docker_cli run --rm \
        -v "${code_path_docker}:/challenge" \
        -v "${data_path_docker}:/challenge/eval_data:ro" \
        -v "${prevalence_path_docker}:/challenge/prevalence_data:ro" \
        "$IMAGE_NAME" \
        python evaluate_model.py \
            -d "/challenge/eval_data/${DEMOGRAPHICS_FILE}" \
            -p "/challenge/prevalence_data/${DEMOGRAPHICS_FILE}" \
            -o "$output_path/${DEMOGRAPHICS_FILE}"
}

dataset_has_labels() {
    local data_dir="$1"
    local demographics_path="$data_dir/$DEMOGRAPHICS_FILE"

    [[ -f "$demographics_path" ]] && head -n 1 "$demographics_path" | grep -q "Cognitive_Impairment"
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

train_smoke() {
    local smoke_data model_smoke
    local feature_cache
    local smoke_data_docker model_smoke_docker feature_cache_docker

    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    model_smoke="$(get_absolute_path ".")/${MODEL_SMOKE_REL}"
    feature_cache="$(get_absolute_path ".")/${FEATURE_CACHE_REL}"
    smoke_data_docker="$(to_docker_path "$smoke_data")"
    model_smoke_docker="$(to_docker_path "$model_smoke")"
    feature_cache_docker="$(to_docker_path "$feature_cache")"

    ensure_directory "$model_smoke"
    ensure_directory "$feature_cache"

    configure_gpu_args

    docker_cli run --rm "${GPU_ARGS[@]}" \
        -v "${smoke_data_docker}:/challenge/training_data:ro" \
        -v "${model_smoke_docker}:/challenge/model" \
        -v "${feature_cache_docker}:/challenge/.feature_cache" \
        "$IMAGE_NAME" \
        python train_model.py -d training_data -m model -v
}

run_full() {
    local run_data model_full out_full
    local feature_cache
    local run_data_docker model_full_docker out_full_docker feature_cache_docker

    run_data="$(get_absolute_path "$RUN_DATA_REL")"
    model_full="$(get_absolute_path "$MODEL_FULL_REL")"
    out_full="$(get_absolute_path ".")/${OUT_FULL_REL}"
    feature_cache="$(get_absolute_path ".")/${FEATURE_CACHE_REL}"
    run_data_docker="$(to_docker_path "$run_data")"
    model_full_docker="$(to_docker_path "$model_full")"
    out_full_docker="$(to_docker_path "$out_full")"
    feature_cache_docker="$(to_docker_path "$feature_cache")"

    ensure_directory "$out_full"
    ensure_directory "$feature_cache"

    configure_gpu_args

    docker_cli run --rm "${GPU_ARGS[@]}" \
        -v "${run_data_docker}:/challenge/holdout_data:ro" \
        -v "${model_full_docker}:/challenge/model:ro" \
        -v "${out_full_docker}:/challenge/holdout_outputs" \
        -v "${feature_cache_docker}:/challenge/.feature_cache" \
        "$IMAGE_NAME" \
        python run_model.py -d holdout_data -m model -o holdout_outputs -v

    if dataset_has_labels "$run_data"; then
        local code_path
        code_path="$(get_absolute_path ".")"

        evaluate_predictions \
            "$code_path" \
            "$run_data" \
            "$out_full" \
            "run-dataset"
    else
        echo "Skipping evaluation..."
    fi
}

run_smoke() {
    local smoke_data model_smoke out_smoke
    local feature_cache
    local smoke_data_docker model_smoke_docker out_smoke_docker feature_cache_docker

    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    model_smoke="$(get_absolute_path "$MODEL_SMOKE_REL")"
    out_smoke="$(get_absolute_path ".")/${OUT_SMOKE_REL}"
    feature_cache="$(get_absolute_path ".")/${FEATURE_CACHE_REL}"
    smoke_data_docker="$(to_docker_path "$smoke_data")"
    model_smoke_docker="$(to_docker_path "$model_smoke")"
    out_smoke_docker="$(to_docker_path "$out_smoke")"
    feature_cache_docker="$(to_docker_path "$feature_cache")"

    ensure_directory "$out_smoke"
    ensure_directory "$feature_cache"

    configure_gpu_args

    docker_cli run --rm "${GPU_ARGS[@]}" \
        -v "${smoke_data_docker}:/challenge/holdout_data:ro" \
        -v "${model_smoke_docker}:/challenge/model:ro" \
        -v "${out_smoke_docker}:/challenge/holdout_outputs" \
        -v "${feature_cache_docker}:/challenge/.feature_cache" \
        "$IMAGE_NAME" \
        python run_model.py -d holdout_data -m model -o holdout_outputs -v

    evaluate_predictions "$smoke_data" "$out_smoke" "smoke"
}

eval_full() {
    local run_data out_full

    run_data="$(get_absolute_path "$RUN_DATA_REL")"
    out_full="$(get_absolute_path "$OUT_FULL_REL")"

    if dataset_has_labels "$run_data"; then
        local code_path
        code_path="$(get_absolute_path ".")"

        evaluate_predictions \
            "$code_path" \
            "$run_data" \
            "$out_full" \
            "run-dataset"
    else
        echo "Skipping evaluation..."
    fi
}

eval_smoke() {
    local smoke_data out_smoke

    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    out_smoke="$(get_absolute_path "$OUT_SMOKE_REL")"

    local code_path
    code_path="$(get_absolute_path ".")"

    evaluate_predictions \
        "$code_path" \
        "$smoke_data" \
        "$out_smoke" \
        "smoke"
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

run_dev() {
    local code_path run_data model_full out_full prevalence_data
    local code_path_docker run_data_docker

    code_path="$(get_absolute_path ".")"
    run_data="$(get_absolute_path "$RUN_DATA_REL")"
    model_full="${code_path}/${MODEL_FULL_REL}"
    out_full="${code_path}/${OUT_FULL_REL}"
    prevalence_data="$(get_absolute_path "$TRAIN_DATA_REL")"
    code_path_docker="$(to_docker_path "$code_path")"
    run_data_docker="$(to_docker_path "$run_data")"

    ensure_directory "$out_full"

    configure_gpu_args

    docker_cli run --rm "${GPU_ARGS[@]}" \
        -v "${code_path_docker}:/challenge" \
        -v "${run_data_docker}:/challenge/holdout_data:ro" \
        "$IMAGE_NAME" \
        python run_model.py -d /challenge/holdout_data -m /challenge/model -o /challenge/outputs -v

    if dataset_has_labels "$run_data"; then
        evaluate_predictions_dev "$code_path" "$run_data" "/challenge/outputs" "$prevalence_data" "development run-dataset"
    else
        echo "Skipping evaluation..."
    fi
}

eval_dev() {
    local code_path run_data prevalence_data

    code_path="$(get_absolute_path ".")"
    run_data="$(get_absolute_path "$RUN_DATA_REL")"
    prevalence_data="$(get_absolute_path "$TRAIN_DATA_REL")"

    if dataset_has_labels "$run_data"; then
        evaluate_predictions_dev "$code_path" "$run_data" "/challenge/outputs" "$prevalence_data" "development run-dataset"
    else
        echo "Skipping evaluation..."
    fi
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
    eval)        eval_full ;;
    eval-smoke)  eval_smoke ;;
    train-dev)   train_dev ;;
    run-dev)     run_dev ;;
    eval-dev)    eval_dev ;;
    clean)       clean_all ;;
    *)
        echo "Invalid command: $COMMAND"
        echo "Valid commands: build, smoke, train, train-smoke, run, run-smoke, eval, eval-smoke, train-dev, run-dev, eval-dev, clean"
        exit 1
        ;;
esac
