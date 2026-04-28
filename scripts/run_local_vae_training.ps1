param(
    [string]$BlenderPath = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
    [string]$FbxDir = "data\houses3k_fbx",
    [string]$ObjDir = "data\houses3k_obj",
    [string]$VoxelDir = "data\houses3k_vox64",
    [string]$OutDir = "outputs\vae3d",
    [int]$Resolution = 64,
    [int]$Epochs = 200,
    [int]$BatchSize = 2
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path "$PSScriptRoot\..")
$env:PYTHONPATH = "src"

Write-Host "[1/3] Convert FBX to OBJ"
python src\convert_fbx_to_obj.py `
    --in_dir $FbxDir `
    --out_dir $ObjDir `
    --blender_path $BlenderPath

Write-Host "[2/3] Build voxel dataset"
python src\build_voxel_dataset.py `
    --mesh_dir $ObjDir `
    --out_dir $VoxelDir `
    --resolution $Resolution `
    --fill_holes `
    --keep_lcc

Write-Host "[3/3] Train 3D VAE"
python src\train_3d_vae.py `
    --data_root $VoxelDir `
    --out_dir $OutDir `
    --device cpu `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --num_workers 0
