<#
.SYNOPSIS
    Starts the pamostore Fraud Detection Engine (Windows).

.PARAMETER Port
    Port to listen on (default 8000).

.PARAMETER BindHost
    Network interface to bind (default 127.0.0.1 = this computer only).
    Use 0.0.0.0 only on a trusted network: the API has no authentication yet.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run.ps1
    powershell -ExecutionPolicy Bypass -File .\run.ps1 -Port 8080
#>
param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$VenvPython = ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: Virtual environment not found. Run setup first:" -ForegroundColor Red
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\setup.ps1"
    exit 1
}

if (-not (Test-Path "models\fraud_model.joblib")) {
    Write-Host "No trained model found - training one now (takes a few seconds)..." -ForegroundColor Yellow
    & $VenvPython generate_synthetic_data.py
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Model training failed." -ForegroundColor Red; exit 1 }
}

Write-Host "Starting server on http://${BindHost}:$Port  (press Ctrl+C to stop)" -ForegroundColor Green
Write-Host "First start downloads the vision model (~600 MB) and can take a few minutes."
& $VenvPython -m uvicorn main:app --host $BindHost --port $Port
