param(
    [string]$ConfigPath = "configs/prior/sample_prior_base.json",
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string[]]$ExtraArgs = @()
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$scriptDir/Invoke-JsonConfig.ps1" -PythonEntry "src/sample_3d_prior.py" -ConfigPath $ConfigPath -PythonExe $PythonExe -ExtraArgs $ExtraArgs
