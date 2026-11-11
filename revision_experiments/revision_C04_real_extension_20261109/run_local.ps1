$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH="$PSScriptRoot\.ops"+[IO.Path]::PathSeparator+$env:PYTHONPATH
$exe='D:\YX\CIM\revision_experiments\.runtime313\Scripts\python.exe'
& $exe -u "$PSScriptRoot\prepare_extension.py"
if($LASTEXITCODE -ne 0){throw 'Validation failed'}
& $exe -u "$PSScriptRoot\run_suite.py" run --plan "$PSScriptRoot\plan.json" --output "$PSScriptRoot\results\extension" --workers 3 --s2v-workers 1
if($LASTEXITCODE -ne 0){throw 'Run failed; completed tasks and checkpoints retained'}
& $exe -u "$PSScriptRoot\summarize_extension.py"
