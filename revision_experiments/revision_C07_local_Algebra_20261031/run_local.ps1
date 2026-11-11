$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$exe='D:\YX\CIM\revision_experiments\.runtime313\Scripts\python.exe'
& $exe -u "$PSScriptRoot\fair_comparison.py" run --live-source D:\YX\CIM\revision_experiments_fast_20261029\results\local_fast --workers 2 --s2v-workers 1
if($LASTEXITCODE -ne 0){throw 'Training failed; see task console.log. Completed tasks/checkpoints retained.'}
& $exe -u "$PSScriptRoot\fair_comparison.py" analyze
