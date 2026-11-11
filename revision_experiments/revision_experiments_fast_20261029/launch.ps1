param(
    [string]$Config = 'validation_synthetic_smoke.json',
    [string]$Output = ('results/run_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $Python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
}
Push-Location $PSScriptRoot
try {
    & $Python -c 'import numpy, scipy, torch'
    if ($LASTEXITCODE -ne 0) { throw 'Run setup.ps1 first or specify launch.ps1 -Python.' }
    & $Python shared_experiment_core.py --config $Config --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; inspect partial output.' }
    & $Python C04_C07_C08_result_summary.py $Output
    if ($LASTEXITCODE -ne 0) { throw 'Summary failed.' }
} finally {
    Pop-Location
}
