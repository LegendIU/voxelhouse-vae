param(
    [string]$ConfigPath = "configs/prior/eval_prior_base.json",
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string[]]$ExtraArgs = @()
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$scriptDir/Invoke-JsonConfig.ps1" -PythonEntry "src/eval_latent_prior.py" -ConfigPath $ConfigPath -PythonExe $PythonExe -ExtraArgs $ExtraArgs
