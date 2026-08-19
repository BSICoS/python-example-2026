param(
    [Parameter(Mandatory=$true)]
    [ValidateSet(
        "build",
        "smoke",
        "train",
        "train-smoke",
        "run-smoke",
        "eval-smoke",
        "train-dev",
        "supplementary",
        "clean"
    )]
    [string]$Command
)

# ============================================
# CONFIGURACIÓN
# ============================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# IMPORTANTE:
# Si tu dataset no está en data/training_set o data/supplementary_set,
# modifica estas rutas.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
$TRAIN_DATA_REL = "data/training_set"
$SMOKE_DATA_REL = "data/training_smoke"
$SUPPLEMENTARY_DATA_REL = "data/supplementary_set"

$IMAGE_NAME = "cinc2026"

$MODEL_FULL_REL = "model"
$MODEL_SMOKE_REL = "model_smoke"
$FEATURE_CACHE_REL = ".feature_cache"

$OUT_SMOKE_REL = "outputs_smoke"
$OUT_SUPPLEMENTARY_REL = "outputs_supplementary"
$DEMOGRAPHICS_FILE = "demographics.csv"

# ============================================
# FUNCIONES AUXILIARES
# ============================================

function Get-AbsolutePath($relativePath) {
    return (Resolve-Path $relativePath).Path
}

function Ensure-Directory($path) {
    if (!(Test-Path $path)) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
    }
}

function Get-DockerGpuArgs {
    $null = docker run --rm --gpus all `
        $IMAGE_NAME `
        python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" `
        2>$null

    if ($LASTEXITCODE -eq 0) {
        Write-Host "CUDA GPU detected. Using GPU."
        return @("--gpus", "all")
    }

    Write-Host "CUDA GPU not available. Using CPU."
    return @()
}

function Invoke-Evaluation($DataPath, $OutputPath, $PrevalencePath, $Label) {
    Write-Host "Evaluating $Label predictions..."
    docker run --rm `
        -v "${DataPath}:/challenge/eval_data:ro" `
        -v "${OutputPath}:/challenge/eval_outputs:ro" `
        -v "${PrevalencePath}:/challenge/prevalence_data:ro" `
        $IMAGE_NAME `
        python evaluate_model.py -d "/challenge/eval_data/$DEMOGRAPHICS_FILE" -o "/challenge/eval_outputs/$DEMOGRAPHICS_FILE" -p "/challenge/prevalence_data/$DEMOGRAPHICS_FILE"
}

# ============================================
# COMANDOS
# ============================================

function Build-Image {
    docker build -t $IMAGE_NAME .
}

function Create-Smoke {
    Write-Host "Creando dataset smoke..."
    powershell -ExecutionPolicy Bypass -File scripts/create_smoke.ps1
}

function Train-Full {

    $FULL_DATA = Get-AbsolutePath $TRAIN_DATA_REL
    $MODEL_FULL = Join-Path (Get-AbsolutePath ".") $MODEL_FULL_REL
    $FEATURE_CACHE = Join-Path (Get-AbsolutePath ".") $FEATURE_CACHE_REL

    Ensure-Directory $MODEL_FULL
    Ensure-Directory $FEATURE_CACHE

    $GPU_ARGS = Get-DockerGpuArgs
    docker run --rm $GPU_ARGS `
        -v "${FULL_DATA}:/challenge/training_data:ro" `
        -v "${MODEL_FULL}:/challenge/model" `
        -v "${FEATURE_CACHE}:/challenge/.feature_cache" `
        $IMAGE_NAME `
        python train_model.py -d training_data -m model -v
}

function Train-Smoke {

    $SMOKE_DATA = Get-AbsolutePath $SMOKE_DATA_REL
    $MODEL_SMOKE = Join-Path (Get-AbsolutePath ".") $MODEL_SMOKE_REL
    $FEATURE_CACHE = Join-Path (Get-AbsolutePath ".") $FEATURE_CACHE_REL

    Ensure-Directory $MODEL_SMOKE
    Ensure-Directory $FEATURE_CACHE

    $GPU_ARGS = Get-DockerGpuArgs
    docker run --rm $GPU_ARGS `
        -v "${SMOKE_DATA}:/challenge/training_data:ro" `
        -v "${MODEL_SMOKE}:/challenge/model" `
        -v "${FEATURE_CACHE}:/challenge/.feature_cache" `
        $IMAGE_NAME `
        python train_model.py -d training_data -m model -v
}

function Run-Smoke {

    $SMOKE_DATA = Get-AbsolutePath $SMOKE_DATA_REL
    $PREVALENCE_DATA = Get-AbsolutePath $TRAIN_DATA_REL
    $MODEL_SMOKE = Get-AbsolutePath $MODEL_SMOKE_REL
    $OUT_SMOKE = Join-Path (Get-AbsolutePath ".") $OUT_SMOKE_REL
    $FEATURE_CACHE = Join-Path (Get-AbsolutePath ".") $FEATURE_CACHE_REL

    Ensure-Directory $OUT_SMOKE
    Ensure-Directory $FEATURE_CACHE

    $GPU_ARGS = Get-DockerGpuArgs
    docker run --rm $GPU_ARGS `
        -v "${SMOKE_DATA}:/challenge/holdout_data:ro" `
        -v "${MODEL_SMOKE}:/challenge/model:ro" `
        -v "${OUT_SMOKE}:/challenge/holdout_outputs" `
        -v "${FEATURE_CACHE}:/challenge/.feature_cache" `
        $IMAGE_NAME `
        python run_model.py -d holdout_data -m model -o holdout_outputs -v

    Invoke-Evaluation $SMOKE_DATA $OUT_SMOKE $PREVALENCE_DATA "smoke"
}

function Eval-Smoke {

    $SMOKE_DATA = Get-AbsolutePath $SMOKE_DATA_REL
    $PREVALENCE_DATA = Get-AbsolutePath $TRAIN_DATA_REL
    $OUT_SMOKE = Get-AbsolutePath $OUT_SMOKE_REL

    Invoke-Evaluation $SMOKE_DATA $OUT_SMOKE $PREVALENCE_DATA "smoke"
}

# ======================
# MODO DESARROLLO (SIN REBUILD, DATASETS COMPLETOS)
# ======================

function Train-Dev {

    $CODE_PATH = Get-AbsolutePath "."
    $FULL_DATA = Get-AbsolutePath $TRAIN_DATA_REL
    $MODEL_FULL = Join-Path $CODE_PATH $MODEL_FULL_REL
    $FEATURE_CACHE = Join-Path $CODE_PATH $FEATURE_CACHE_REL

    Ensure-Directory $MODEL_FULL
    Ensure-Directory $FEATURE_CACHE

    $GPU_ARGS = Get-DockerGpuArgs
    docker run --rm $GPU_ARGS `
        -v "${CODE_PATH}:/challenge" `
        -v "${FULL_DATA}:/challenge/training_data:ro" `
        -v "${MODEL_FULL}:/challenge/model" `
        $IMAGE_NAME `
        python train_model.py -d training_data -m model -v
}

function Run-Supplementary {

    $MODEL_FILE = Join-Path $MODEL_FULL_REL "model.sav"
    if (!(Test-Path $MODEL_FILE)) {
        throw "A trained model is required at $MODEL_FILE. Run '.\run.ps1 train' first."
    }

    $SUPPLEMENTARY_DATA = Get-AbsolutePath $SUPPLEMENTARY_DATA_REL
    $MODEL_FULL = Get-AbsolutePath $MODEL_FULL_REL
    $OUT_SUPPLEMENTARY = Join-Path (Get-AbsolutePath ".") $OUT_SUPPLEMENTARY_REL
    $FEATURE_CACHE = Join-Path (Get-AbsolutePath ".") $FEATURE_CACHE_REL

    Ensure-Directory $OUT_SUPPLEMENTARY
    Ensure-Directory $FEATURE_CACHE

    $GPU_ARGS = Get-DockerGpuArgs
    docker run --rm $GPU_ARGS `
        -v "${SUPPLEMENTARY_DATA}:/challenge/supplementary_data:ro" `
        -v "${MODEL_FULL}:/challenge/model:ro" `
        -v "${OUT_SUPPLEMENTARY}:/challenge/supplementary_outputs" `
        -v "${FEATURE_CACHE}:/challenge/.feature_cache" `
        $IMAGE_NAME `
        python run_model.py -d supplementary_data -m model -o supplementary_outputs -v
}

function Clean-All {

    Remove-Item -Recurse -Force $MODEL_FULL_REL -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $MODEL_SMOKE_REL -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $OUT_SMOKE_REL -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $OUT_SUPPLEMENTARY_REL -ErrorAction SilentlyContinue

    Write-Host "Modelos y outputs eliminados."
}

# ============================================
# SWITCH PRINCIPAL
# ============================================

switch ($Command) {

    "build"       { Build-Image }
    "smoke"       { Create-Smoke }
    "train"       { Train-Full }
    "train-smoke" { Train-Smoke }
    "run-smoke"   { Run-Smoke }
    "eval-smoke"  { Eval-Smoke }
    "train-dev"   { Train-Dev }
    "supplementary" { Run-Supplementary }
    "clean"       { Clean-All }

}
