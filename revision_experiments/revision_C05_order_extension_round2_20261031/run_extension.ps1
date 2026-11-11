$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$fix = Join-Path $PSScriptRoot '.ops'
if ($env:PYTHONPATH) { $env:PYTHONPATH = $fix + [IO.Path]::PathSeparator + $env:PYTHONPATH } else { $env:PYTHONPATH = $fix }
$exe = 'D:/YX/CIM/revision_experiments/.runtime313/Scripts/python.exe'
foreach ($networkName in @('Music-Rev','NDC-classes','high-school_2012')) {
    Write-Output "Starting fixed-cohort network: $networkName"
    & $exe -u "$PSScriptRoot/run_suite.py" run --plan "$PSScriptRoot/plan_order_extension.json" --task-file "$PSScriptRoot/tasks_$networkName.json" --output "$PSScriptRoot/results/extension" --workers 3 --s2v-workers 1
    if ($LASTEXITCODE -ne 0) { throw "Network $networkName stopped. Completed results and checkpoints remain saved." }
    & $exe "$PSScriptRoot/summarize_extension.py" --allow-partial
    if ($LASTEXITCODE -ne 0) { throw 'Summary failed; inspect error, raw results retained.' }
}
& $exe "$PSScriptRoot/summarize_extension.py"
if ($LASTEXITCODE -ne 0) { throw 'Final validation failed; raw results retained.' }
