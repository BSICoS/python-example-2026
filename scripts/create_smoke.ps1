# ============================================
# Create smoke training dataset
# ============================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# IMPORTANT:
# Each team member must modify this path to
# match their local dataset location.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
$FULL_DATA_PATH = "data/training_set"  # <-- CHANGE THIS IF NEEDED

$SMOKE_PATH = "data/training_smoke"
$N_RECORDS = 5

Write-Host "Creating smoke dataset..."
Write-Host "Source: $FULL_DATA_PATH"
Write-Host "Destination: $SMOKE_PATH"

Remove-Item -Recurse -Force $SMOKE_PATH -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $SMOKE_PATH | Out-Null

# Copy demographics
Copy-Item "$FULL_DATA_PATH/demographics.csv" "$SMOKE_PATH/demographics.csv"

# Select first N EDF files
$edfs = Get-ChildItem "$FULL_DATA_PATH/physiological_data" -Recurse -Filter *.edf |
        Sort-Object FullName |
        Select-Object -First $N_RECORDS

foreach ($f in $edfs) {
    $rel = $f.FullName.Substring((Resolve-Path $FULL_DATA_PATH).Path.Length).TrimStart('\')
    $target = Join-Path $SMOKE_PATH $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    Copy-Item $f.FullName $target
}

# Copy full annotation folders (simpler and robust)
Copy-Item "$FULL_DATA_PATH/algorithmic_annotations" "$SMOKE_PATH/algorithmic_annotations" -Recurse -ErrorAction SilentlyContinue
Copy-Item "$FULL_DATA_PATH/human_annotations" "$SMOKE_PATH/human_annotations" -Recurse -ErrorAction SilentlyContinue

Write-Host "Smoke dataset created successfully."