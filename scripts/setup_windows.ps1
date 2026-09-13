$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'X64') { throw 'Windows x86-64 required' }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/0.12.13/install.ps1 | Invoke-Expression
    $env:PATH = "$HOME\.local\bin;$env:PATH"
}
uv python install 3.11.16
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed' }
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    uv venv --python 3.11.16 --seed .venv
    if ($LASTEXITCODE -ne 0) { throw 'venv failed' }
}
& .venv\Scripts\python.exe scripts/setup.py
if ($LASTEXITCODE -ne 0) { throw 'Setup failed' }
