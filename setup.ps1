<#
.SYNOPSIS
    One-command setup for the pamostore Fraud Detection Engine (Windows).

.DESCRIPTION
    1. Verifies Python 3.10+ is installed.
    2. Creates an isolated virtual environment in .venv
    3. Installs all dependencies from requirements.txt
    4. Pre-trains the cold-start fraud model (models/fraud_model.joblib)

.PARAMETER Retrain
    Force re-training of the fraud model even if one already exists.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup.ps1
#>
param(
    [switch]$Retrain
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$MinPythonMajor = 3
$MinPythonMinor = 10
$VenvDir = ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$ModelFile = "models\fraud_model.joblib"

<#
.SYNOPSIS
    Prints a formatted step header.
.PARAMETER Message
    Text describing the current setup step.
#>
function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

<#
.SYNOPSIS
    Stops the script with a friendly error message.
.PARAMETER Message
    Explanation of what went wrong and how to fix it.
#>
function Fail([string]$Message) {
    Write-Host ""
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

<#
.SYNOPSIS
    Finds a Python 3.10+ interpreter on this machine.
.OUTPUTS
    String command name ("python" or "py"), or $null if none is suitable.
#>
function Find-Python {
    foreach ($candidate in @("python", "py")) {
        if (-not (Get-Command $candidate -ErrorAction SilentlyContinue)) { continue }
        $version = & $candidate -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $version) { continue }
        $parts = $version.Trim().Split(".")
        $major = [int]$parts[0]; $minor = [int]$parts[1]
        if ($major -gt $MinPythonMajor -or ($major -eq $MinPythonMajor -and $minor -ge $MinPythonMinor)) {
            return $candidate
        }
    }
    return $null
}

Write-Step "Checking for Python $MinPythonMajor.$MinPythonMinor or newer"
$Python = Find-Python
if (-not $Python) {
    Fail "Python $MinPythonMajor.$MinPythonMinor+ was not found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then re-run this script."
}
Write-Host "Using: $(& $Python --version)"

Write-Step "Creating virtual environment in $VenvDir"
if (Test-Path $VenvPython) {
    Write-Host "Virtual environment already exists - reusing it."
} else {
    & $Python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment." }
}

Write-Step "Installing dependencies (first run downloads ~1 GB, please be patient)"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Fail "Could not upgrade pip. Check your internet connection." }
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed. Check your internet connection and free disk space (~3 GB needed)." }

Write-Step "Preparing the fraud model"
if ((Test-Path $ModelFile) -and -not $Retrain) {
    Write-Host "Model already exists at $ModelFile - skipping training (use -Retrain to rebuild)."
} else {
    & $VenvPython generate_synthetic_data.py
    if ($LASTEXITCODE -ne 0) { Fail "Model pre-training failed. See the log output above." }
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Start the server with:   powershell -ExecutionPolicy Bypass -File .\run.ps1"
Write-Host "Then open:               http://127.0.0.1:8000"
