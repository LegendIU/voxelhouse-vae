param(
    [Parameter(Mandatory = $true)][string]$PythonEntry,
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [string]$PythonExe = "python",
    [string[]]$ExtraArgs = @()
)

if (-not (Test-Path $ConfigPath)) {
    throw "Config file not found: $ConfigPath"
}

$config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
$argsList = @($PythonEntry)

foreach ($prop in $config.PSObject.Properties) {
    $name = "--$($prop.Name)"
    $value = $prop.Value

    if ($null -eq $value) {
        continue
    }

    if ($value -is [bool]) {
        if ($value) {
            $argsList += $name
        }
        continue
    }

    if ($value -is [System.Array]) {
        foreach ($item in $value) {
            $argsList += $name
            $argsList += [string]$item
        }
        continue
    }

    $argsList += $name
    $argsList += [string]$value
}

if ($ExtraArgs.Count -gt 0) {
    $argsList += $ExtraArgs
}

Write-Host "Running: $PythonExe $($argsList -join ' ')"
& $PythonExe @argsList
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
