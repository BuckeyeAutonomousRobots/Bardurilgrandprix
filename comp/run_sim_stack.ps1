param(
    [switch]$NoLaunch,
    [switch]$Reset,
    [switch]$VisionPrimary,
    [switch]$ShowVision,
    [switch]$KeepPrevious,
    [double]$MaxSeconds = 0,
    [int]$WaitSeconds = 30,
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$simRoot = Join-Path $root "AIGP_3364"
$simExe = Join-Path $simRoot "FlightSim.exe"

function Resolve-StackPython {
    $venvPython = Join-Path $root ".venv\Scripts\python.exe"
    $legacyVenv = Join-Path $root "pilot\.venv\Scripts\python.exe"
    $candidates = @()
    if (Test-Path $venvPython) {
        $candidates += $venvPython
    }
    if (Test-Path $legacyVenv) {
        $candidates += $legacyVenv
    }
    $candidates += "python"

    foreach ($candidate in $candidates) {
        try {
            & $candidate -c "import pymavlink, numpy, cv2, yaml" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {}
    }

    throw "No working Python with stack deps. Run: pip install -r requirements.txt"
}

function Stop-PreviousSimStack {
    param([string]$RootDir)
    $lockPath = Join-Path $RootDir "logs\sim_stack.active.json"
    if (-not (Test-Path $lockPath)) {
        return
    }
    try {
        $payload = Get-Content $lockPath -Raw | ConvertFrom-Json
        $lockPid = [int]$payload.pid
        $logPath = [string]$payload.log_path
    } catch {
        Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
        return
    }
    if ($lockPid -le 0) {
        Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
        return
    }
    $proc = Get-Process -Id $lockPid -ErrorAction SilentlyContinue
    if ($null -eq $proc) {
        Write-Host "[SIM-STACK] Clearing stale active lock (pid=$lockPid not running)."
        Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
        return
    }
    Write-Host "[SIM-STACK] Stopping previous sim stack (pid=$lockPid, log=$logPath)..."
    Stop-Process -Id $lockPid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 400
    Remove-Item $lockPath -Force -ErrorAction SilentlyContinue
}

if (-not $NoLaunch) {
    if (-not (Test-Path $simExe)) {
        throw "Simulator executable not found: $simExe"
    }
    $existingSim = Get-Process -Name "FlightSim" -ErrorAction SilentlyContinue
    if ($existingSim) {
        Write-Host "[SIM-STACK] Simulator already running (PID: $($existingSim.Id)). Skipping launch."
    } else {
        Write-Host "[SIM-STACK] Launching simulator..."
        Start-Process -FilePath $simExe -WorkingDirectory $simRoot | Out-Null
    }
    Write-Host "[SIM-STACK] Waiting $WaitSeconds seconds for simulator load..."
    Start-Sleep -Seconds $WaitSeconds
}

$python = Resolve-StackPython
if (-not $KeepPrevious) {
    Stop-PreviousSimStack -RootDir $root
}
$argsList = @("-m", "src.main")
if ($VisionPrimary) {
    $argsList += "--sim-config"
    $argsList += "config/sim_comp.yaml"
}
if ($MaxSeconds -gt 0) {
    $argsList += "--max-seconds"
    $argsList += "$MaxSeconds"
}
if ($LogPath) {
    $argsList += "--log-path"
    $argsList += $LogPath
}
if ($ShowVision) {
    $argsList += "--show-vision"
}

Write-Host "[SIM-STACK] Starting modular sim gate-racing stack (src.main)..."
if ($VisionPrimary) {
    Write-Host "[SIM-STACK] Mode: vision-primary (comp-legal navigation)"
}
if ($ShowVision) {
    Write-Host "[SIM-STACK] Live camera preview enabled (press q in window to quit)"
}
Push-Location $root
try {
    & $python @argsList
} finally {
    Pop-Location
}
