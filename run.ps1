param(
    [Parameter(Mandatory=$true)]
    [ValidateSet(
        "build",
        "smoke",
        "train",
        "train-smoke",
        "run",
        "run-smoke",
        "eval",
        "eval-smoke",
        "train-dev",
        "run-dev",
        "eval-dev",
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

function Invoke-Evaluation($DataPath, $OutputPath, $Label) {
    Write-Host "Evaluating $Label predictions..."
    docker run --rm `
        -v "${DataPath}:/challenge/eval_data:ro" `
        -v "${OutputPath}:/challenge/eval_outputs:ro" `
        $IMAGE_NAME `
        python evaluate_model.py -d "/challenge/eval_data/$DEMOGRAPHICS_FILE" -o "/challenge/eval_outputs/$DEMOGRAPHICS_FILE"
}

function Invoke-EvaluationDev($CodePath, $DataPath, $OutputPath, $Label) {
    Write-Host "Evaluating $Label predictions..."
    docker run --rm `
        -v "${CodePath}:/challenge" `
        -v "${DataPath}:/challenge/eval_data:ro" `
        $IMAGE_NAME `
        python evaluate_model.py -d "/challenge/eval_data/$DEMOGRAPHICS_FILE" -o "$OutputPath/$DEMOGRAPHICS_FILE"
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

    Invoke-Evaluation $FULL_DATA $OUT_FULL "full-dataset"
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

    Invoke-Evaluation $SMOKE_DATA $OUT_SMOKE "smoke"
}

function Eval-Full {

    $FULL_DATA = Get-AbsolutePath $FULL_DATA_REL
    $OUT_FULL = Get-AbsolutePath $OUT_FULL_REL

    Invoke-Evaluation $FULL_DATA $OUT_FULL "full-dataset"
}

function Eval-Smoke {

    $SMOKE_DATA = Get-AbsolutePath $SMOKE_DATA_REL
    $OUT_SMOKE = Get-AbsolutePath $OUT_SMOKE_REL

    Invoke-Evaluation $SMOKE_DATA $OUT_SMOKE "smoke"
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

    Invoke-EvaluationDev $CODE_PATH $SMOKE_DATA "/challenge/holdout_outputs" "development smoke"
}

function Eval-Dev {

    $CODE_PATH = Get-AbsolutePath "."
    $SMOKE_DATA = Get-AbsolutePath $SMOKE_DATA_REL

    Invoke-EvaluationDev $CODE_PATH $SMOKE_DATA "/challenge/holdout_outputs" "development smoke"
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
    "eval"        { Eval-Full }
    "eval-smoke"  { Eval-Smoke }
    "train-dev"   { Train-Dev }
    "run-dev"     { Run-Dev }
    "eval-dev"    { Eval-Dev }
    "clean"       { Clean-All }

}