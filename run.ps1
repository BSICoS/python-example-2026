param(
    [Parameter(Mandatory=$true)]
    [ValidateSet(
        "build",
        "smoke",
        "train",
        "train-smoke",
        "run",
        "run-smoke",
        "train-dev",
        "run-dev",
        "clean"
    )]
    [string]$Command
)

# ============================================
# CONFIGURACIÓN
# ============================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# IMPORTANTE:
# Si tu dataset no está en data/training_set,
# modifica esta ruta.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
$FULL_DATA_REL = "data/training_set"
$SMOKE_DATA_REL = "data/training_smoke"

$IMAGE_NAME = "cinc2026"

$MODEL_FULL_REL = "model"
$MODEL_SMOKE_REL = "model_smoke"

$OUT_FULL_REL = "outputs"
$OUT_SMOKE_REL = "outputs_smoke"

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

    $FULL_DATA = Get-AbsolutePath $FULL_DATA_REL
    $MODEL_FULL = Join-Path (Get-AbsolutePath ".") $MODEL_FULL_REL

    Ensure-Directory $MODEL_FULL

    docker run --rm `
        -v "${FULL_DATA}:/challenge/training_data:ro" `
        -v "${MODEL_FULL}:/challenge/model" `
        $IMAGE_NAME `
        python train_model.py -d training_data -m model -v
}

function Train-Smoke {

    $SMOKE_DATA = Get-AbsolutePath $SMOKE_DATA_REL
    $MODEL_SMOKE = Join-Path (Get-AbsolutePath ".") $MODEL_SMOKE_REL

    Ensure-Directory $MODEL_SMOKE

    docker run --rm `
        -v "${SMOKE_DATA}:/challenge/training_data:ro" `
        -v "${MODEL_SMOKE}:/challenge/model" `
        $IMAGE_NAME `
        python train_model.py -d training_data -m model -v
}

function Run-Full {

    $FULL_DATA = Get-AbsolutePath $FULL_DATA_REL
    $MODEL_FULL = Get-AbsolutePath $MODEL_FULL_REL
    $OUT_FULL = Join-Path (Get-AbsolutePath ".") $OUT_FULL_REL

    Ensure-Directory $OUT_FULL

    docker run --rm `
        -v "${FULL_DATA}:/challenge/holdout_data:ro" `
        -v "${MODEL_FULL}:/challenge/model:ro" `
        -v "${OUT_FULL}:/challenge/holdout_outputs" `
        $IMAGE_NAME `
        python run_model.py -d holdout_data -m model -o holdout_outputs -v
}

function Run-Smoke {

    $SMOKE_DATA = Get-AbsolutePath $SMOKE_DATA_REL
    $MODEL_SMOKE = Get-AbsolutePath $MODEL_SMOKE_REL
    $OUT_SMOKE = Join-Path (Get-AbsolutePath ".") $OUT_SMOKE_REL

    Ensure-Directory $OUT_SMOKE

    docker run --rm `
        -v "${SMOKE_DATA}:/challenge/holdout_data:ro" `
        -v "${MODEL_SMOKE}:/challenge/model:ro" `
        -v "${OUT_SMOKE}:/challenge/holdout_outputs" `
        $IMAGE_NAME `
        python run_model.py -d holdout_data -m model -o holdout_outputs -v
}

# ======================
# MODO DESARROLLO (SIN REBUILD)
# ======================

function Train-Dev {

    $CODE_PATH = Get-AbsolutePath "."
    $SMOKE_DATA = Get-AbsolutePath $SMOKE_DATA_REL
    $MODEL_SMOKE = Join-Path $CODE_PATH $MODEL_SMOKE_REL

    Ensure-Directory $MODEL_SMOKE

    docker run --rm `
        -v "${CODE_PATH}:/challenge" `
        -v "${SMOKE_DATA}:/challenge/training_data:ro" `
        -v "${MODEL_SMOKE}:/challenge/model" `
        $IMAGE_NAME `
        python train_model.py -d training_data -m model -v
}

function Run-Dev {

    $CODE_PATH = Get-AbsolutePath "."
    $SMOKE_DATA = Get-AbsolutePath $SMOKE_DATA_REL
    $MODEL_SMOKE = Get-AbsolutePath $MODEL_SMOKE_REL
    $OUT_SMOKE = Join-Path $CODE_PATH $OUT_SMOKE_REL

    Ensure-Directory $OUT_SMOKE

    docker run --rm `
        -v "${CODE_PATH}:/challenge" `
        -v "${SMOKE_DATA}:/challenge/holdout_data:ro" `
        -v "${MODEL_SMOKE}:/challenge/model:ro" `
        -v "${OUT_SMOKE}:/challenge/holdout_outputs" `
        $IMAGE_NAME `
        python run_model.py -d holdout_data -m model -o holdout_outputs -v
}

function Clean-All {

    Remove-Item -Recurse -Force $MODEL_FULL_REL -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $MODEL_SMOKE_REL -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $OUT_FULL_REL -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $OUT_SMOKE_REL -ErrorAction SilentlyContinue

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
    "run"         { Run-Full }
    "run-smoke"   { Run-Smoke }
    "train-dev"   { Train-Dev }
    "run-dev"     { Run-Dev }
    "clean"       { Clean-All }

}