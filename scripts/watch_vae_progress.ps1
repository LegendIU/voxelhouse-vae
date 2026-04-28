param(
    [string]$RunDir = "",
    [int]$TotalEpochs = 200,
    [int]$RefreshSeconds = 10
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..")

if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $latest = Get-ChildItem -Path "outputs\vae3d_gpu" -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No run directory found under outputs\vae3d_gpu"
    }
    $RunDir = $latest.FullName
}

$metrics = Join-Path $RunDir "metrics.csv"
$start = Get-Date

while ($true) {
    Clear-Host
    Write-Host "Run: $RunDir"

    if (Test-Path $metrics) {
        $rows = Import-Csv $metrics
        if ($rows.Count -gt 0) {
            $last = $rows[-1]
            $epoch = [int]$last.epoch
            $pct = [math]::Min(100, [math]::Round(($epoch / $TotalEpochs) * 100, 1))
            $elapsed = (Get-Date) - $start
            $eta = "calculating"
            if ($epoch -gt 0) {
                $totalSeconds = $elapsed.TotalSeconds / $epoch * $TotalEpochs
                $etaSpan = [TimeSpan]::FromSeconds([math]::Max(0, $totalSeconds - $elapsed.TotalSeconds))
                $eta = "{0:hh\:mm\:ss}" -f $etaSpan
            }

            $barWidth = 40
            $filled = [int][math]::Round($barWidth * $epoch / $TotalEpochs)
            $bar = ("#" * $filled).PadRight($barWidth, "-")

            Write-Host ("[{0}] {1}/{2} epochs ({3}%) ETA {4}" -f $bar, $epoch, $TotalEpochs, $pct, $eta)
            Write-Host ("val_iou={0} best_iou={1} train_loss={2} val_loss={3}" -f $last.val_iou, $last.best_val_iou_so_far, $last.train_loss, $last.val_loss)
        } else {
            Write-Host "metrics.csv exists, waiting for first epoch..."
        }
    } else {
        Write-Host "Waiting for metrics.csv..."
    }

    try {
        nvidia-smi --query-gpu=name,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader
    } catch {
        Write-Host "nvidia-smi unavailable"
    }

    Start-Sleep -Seconds $RefreshSeconds
}
