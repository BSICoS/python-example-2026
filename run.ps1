param(
    [Parameter(Mandatory=$true)]
    [ValidateSet(
        "build",
        "train",
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
$SUPPLEMENTARY_DATA_REL = "data/supplementary_set"

$IMAGE_NAME = "cinc2026"

$MODEL_FULL_REL = "model"
$FEATURE_CACHE_REL = ".feature_cache"

$OUT_SUPPLEMENTARY_REL = "outputs_supplementary"

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

# ============================================
# COMANDOS
# ============================================

function Build-Image {
    docker build -t $IMAGE_NAME .
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
    Remove-Item -Recurse -Force $OUT_SUPPLEMENTARY_REL -ErrorAction SilentlyContinue

    Write-Host "Modelos y outputs eliminados."
}

# ============================================
# SWITCH PRINCIPAL
# ============================================

switch ($Command) {

    "build"       { Build-Image }
    "train"       { Train-Full }
    "train-dev"   { Train-Dev }
    "supplementary" { Run-Supplementary }
    "clean"       { Clean-All }

}
