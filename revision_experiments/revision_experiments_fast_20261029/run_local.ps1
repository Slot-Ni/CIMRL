$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot/.runtime313/Scripts/python.exe" -u "$PSScriptRoot/run_suite.py" run --output "$PSScriptRoot/results/local"
if ($LASTEXITCODE -ne 0) { throw 'Run stopped; inspect the error. Completed tasks and training checkpoints remain saved.' }
