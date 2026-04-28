param(
    [string]$DataRoot = "data\houses3k_vox64",
    [string]$OutDir = "outputs\vae3d_gpu",
    [string]$ExportDir = "outputs\vae3d_gpu_exports",
    [int]$Epochs = 200,
    [int]$BatchSize = 4,
    [int]$NumWorkers = 4,
    [int]$NSamples = 64,
    [int]$NMeshes = 32,
    [switch]$Amp
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path "$PSScriptRoot\..")
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$Python = ".\.venv\Scripts\python.exe"

Write-Host "[0/3] Verify CUDA"
& $Python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

$trainArgs = @(
    "src\train_3d_vae.py",
    "--data_root", $DataRoot,
    "--out_dir", $OutDir,
    "--device", "cuda",
    "--epochs", "$Epochs",
    "--batch_size", "$BatchSize",
    "--num_workers", "$NumWorkers",
    "--pin_memory",
    "--save_every", "5"
)

if ($Amp) {
    $trainArgs += "--amp"
}

Write-Host "[1/3] Train 3D VAE on CUDA"
& $Python @trainArgs

$latestRun = Get-ChildItem -Path $OutDir -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $latestRun) {
    throw "No training run directory found under $OutDir"
}

$ckpt = Join-Path $latestRun.FullName "best.pt"
if (-not (Test-Path $ckpt)) {
    $ckpt = Join-Path $latestRun.FullName "last.pt"
}

Write-Host "[2/3] Export PNG projections and OBJ meshes from $ckpt"

& $Python src\infer_3d.py `
    --ckpt $ckpt `
    --out_dir (Join-Path $ExportDir "reconstruct") `
    --sample_mode reconstruct `
    --data_root $DataRoot `
    --split test `
    --n_samples $NSamples `
    --n_meshes $NMeshes `
    --device cuda `
    --export_projections `
    --export_meshes `
    --save_individual_projections

& $Python src\infer_3d.py `
    --ckpt $ckpt `
    --out_dir (Join-Path $ExportDir "posterior_noise") `
    --sample_mode posterior_noise `
    --data_root $DataRoot `
    --split test `
    --n_samples $NSamples `
    --n_meshes $NMeshes `
    --device cuda `
    --export_projections `
    --export_meshes `
    --save_individual_projections

& $Python src\infer_3d.py `
    --ckpt $ckpt `
    --out_dir (Join-Path $ExportDir "prior") `
    --sample_mode prior `
    --n_samples $NSamples `
    --n_meshes $NMeshes `
    --device cuda `
    --export_projections `
    --export_meshes `
    --save_individual_projections

& $Python src\infer_3d.py `
    --ckpt $ckpt `
    --out_dir (Join-Path $ExportDir "interpolate") `
    --sample_mode interpolate `
    --data_root $DataRoot `
    --split test `
    --interp_steps 16 `
    --n_samples 16 `
    --n_meshes 16 `
    --device cuda `
    --export_projections `
    --export_meshes `
    --save_individual_projections

Write-Host "[3/3] Done"
Write-Host "Run: $($latestRun.FullName)"
Write-Host "Exports: $((Resolve-Path $ExportDir).Path)"
