$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$checkpointFix = Join-Path $PSScriptRoot '.ops'
if ($env:PYTHONPATH) { $env:PYTHONPATH = $checkpointFix + [IO.Path]::PathSeparator + $env:PYTHONPATH } else { $env:PYTHONPATH = $checkpointFix }
$exe = 'D:/YX/CIM/revision_experiments/.runtime313/Scripts/python.exe'
& $exe -u "$PSScriptRoot/run_suite.py" run --plan "$PSScriptRoot/plan.json" --task-file "$PSScriptRoot/tasks_local.json" --output "$PSScriptRoot/results/local_fast" --workers 2 --s2v-workers 1
if ($LASTEXITCODE -ne 0) { throw 'Fast run stopped. Check task console.log; results and checkpoints retained.' }
