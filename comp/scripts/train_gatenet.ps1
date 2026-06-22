param(
    [int]$Epochs = 40,
    [int]$Samples = 3000,
    [string]$Output = "models/gate_net.pth"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

Push-Location $root
try {
    & $python -m src.tools.train_gate_net `
        --epochs $Epochs `
        --samples $Samples `
        --output $Output
} finally {
    Pop-Location
}
