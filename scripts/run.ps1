param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("build","smoke","train","train-smoke","run","run-smoke","clean")]
    [string]$Command
)

# ============================================
# CONFIGURATION
# ============================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# IMPORTANTE:
# Si tu dataset no está en data/training_set,
# modifica esta ruta.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
$FULL_DATA="data/training_set"

$SMOKE_DATA="data/training_smoke"

$IMAGE_NAME="cinc2026"
$MODEL_FULL="model"
$MODEL_SMOKE="model_smoke"
$OUT_FULL="outputs"
$OUT_SMOKE="outputs_smoke"

# ============================================
# FUNCTIONS
# ============================================

function Build-Image {
    docker build -t $IMAGE_NAME .
}

function Create-Smoke {
    Write-Host "Creating smoke dataset..."
    powershell -ExecutionPolicy Bypass -File scripts/create_smoke.ps1
}

function Train-Full {
    New-Item -ItemType Directory -Force -Path $MODEL_FULL | Out-Null

    docker run --rm `
        -v "${FULL_DATA}:/challenge/training_data:ro" `
        -v "${PWD}/${MODEL_FULL}:/challenge/model" `
        $IMAGE_NAME `
        python train_model.py -d training_data -m model -v
}

function Train-Smoke {
    New-Item -ItemType Directory -Force -Path $MODEL_SMOKE | Out-Null

    docker run --rm `
        -v "${SMOKE_DATA}:/challenge/training_data:ro" `
        -v "${PWD}/${MODEL_SMOKE}:/challenge/model" `
        $IMAGE_NAME `
        python train_model.py -d training_data -m model -v
}

function Run-Full {
    New-Item -ItemType Directory -Force -Path $OUT_FULL | Out-Null

    docker run --rm `
        -v "${FULL_DATA}:/challenge/holdout_data:ro" `
        -v "${PWD}/${MODEL_FULL}:/challenge/model:ro" `
        -v "${PWD}/${OUT_FULL}:/challenge/holdout_outputs" `
        $IMAGE_NAME `
        python run_model.py -d holdout_data -m model -o holdout_outputs -v
}

function Run-Smoke {
    New-Item -ItemType Directory -Force -Path $OUT_SMOKE | Out-Null

    docker run --rm `
        -v "${SMOKE_DATA}:/challenge/holdout_data:ro" `
        -v "${PWD}/${MODEL_SMOKE}:/challenge/model:ro" `
        -v "${PWD}/${OUT_SMOKE}:/challenge/holdout_outputs" `
        $IMAGE_NAME `
        python run_model.py -d holdout_data -m model -o holdout_outputs -v
}

function Clean-All {
    Remove-Item -Recurse -Force $MODEL_FULL -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $MODEL_SMOKE -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $OUT_FULL -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $OUT_SMOKE -ErrorAction SilentlyContinue
    Write-Host "Cleaned model and output folders."
}

# ============================================
# COMMAND SWITCH
# ============================================

switch ($Command) {

    "build" {
        Build-Image
    }

    "smoke" {
        Create-Smoke
    }

    "train" {
        Train-Full
    }

    "train-smoke" {
        Train-Smoke
    }

    "run" {
        Run-Full
    }

    "run-smoke" {
        Run-Smoke
    }

    "clean" {
        Clean-All
    }

}