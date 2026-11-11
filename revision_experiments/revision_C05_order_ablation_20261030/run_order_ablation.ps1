$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$fix = Join-Path $PSScriptRoot '.ops'
if ($env:PYTHONPATH) { $env:PYTHONPATH = $fix + [IO.Path]::PathSeparator + $env:PYTHONPATH } else { $env:PYTHONPATH = $fix }
$exe = 'D:/YX/CIM/revision_experiments/.runtime313/Scripts/python.exe'
& $exe -u "$PSScriptRoot/run_suite.py" run --plan "$PSScriptRoot/plan_order_ablation.json" --output "$PSScriptRoot/results/first_order" --workers 3 --s2v-workers 1
if ($LASTEXITCODE -ne 0) { throw 'Order ablation stopped. Read task console.log; checkpoints and results retained.' }
& $exe "$PSScriptRoot/summarize_order_ablation.py"
if ($LASTEXITCODE -ne 0) { throw 'Runs completed but summary failed; raw results remain saved.' }
