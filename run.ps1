param([string]$Video = '', [string]$Label = 'Book', [switch]$Show)
$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    & py -3.12 -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required' }
    & $pythonExe -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
}
$runArgs = @((Join-Path $PSScriptRoot 'main.py'))
if (-not $Video) {
    $runArgs += (Join-Path $PSScriptRoot 'data/my_video.mp4')
    $runArgs += @('--quad','60','131','687','130','734','1049','34','1060')
} else {
    $runArgs += $Video
    $runArgs += '--select'
}
$runArgs += @('--label', $Label, '--output', (Join-Path $PSScriptRoot 'results'))
if ($Show) { $runArgs += '--show' }
& $pythonExe @runArgs
exit $LASTEXITCODE
