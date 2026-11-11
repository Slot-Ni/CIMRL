param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$envDir = Join-Path $PSScriptRoot '.venv'
if (Test-Path -LiteralPath $envDir) { throw 'Existing .venv; use it or choose a fresh package directory.' }
& $Python -c 'import sys; assert sys.version_info[:2] == (3,12) and sys.maxsize > 2**32'
if ($LASTEXITCODE -ne 0) { throw 'Use 64-bit Python 3.12.' }
& $Python -m venv $envDir
if ($LASTEXITCODE -ne 0) { throw 'venv failed' }
$exe = Join-Path $envDir 'Scripts/python.exe'
& $exe -m pip install 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
& $exe -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installation failed; keep logs and do not claim success.' }
& $exe -m ipykernel install --user --name ins-revision --display-name 'INS Revision (Python 3.12)'
