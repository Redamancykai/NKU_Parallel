$ErrorActionPreference = "Stop"

$baseUrl = "https://developer.download.nvidia.com/compute/cuda/redist"
$installRoot = Join-Path $PSScriptRoot "cuda-redist"
$archiveRoot = Join-Path $installRoot "archives"

$packages = @(
    "cccl/windows-x86_64/cccl-windows-x86_64-13.3.3.3.1-archive.zip",
    "cuda_crt/windows-x86_64/cuda_crt-windows-x86_64-13.3.33-archive.zip",
    "cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-13.3.29-archive.zip",
    "cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-13.3.33-archive.zip",
    "libnvfatbin/windows-x86_64/libnvfatbin-windows-x86_64-13.3.29-archive.zip",
    "libnvptxcompiler/windows-x86_64/libnvptxcompiler-windows-x86_64-13.3.33-archive.zip",
    "libnvvm/windows-x86_64/libnvvm-windows-x86_64-13.3.33-archive.zip"
)

New-Item -ItemType Directory -Force -Path $installRoot | Out-Null
New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null

foreach ($package in $packages) {
    $fileName = Split-Path $package -Leaf
    $archivePath = Join-Path $archiveRoot $fileName
    $url = "$baseUrl/$package"

    if (!(Test-Path $archivePath)) {
        Write-Host "Downloading $fileName"
        curl.exe -L $url -o $archivePath
    }

    Write-Host "Extracting $fileName"
    Expand-Archive -Force -Path $archivePath -DestinationPath $installRoot
}

$binDirs = Get-ChildItem -Path $installRoot -Recurse -Directory |
    Where-Object { Test-Path (Join-Path $_.FullName "nvcc.exe") }

if ($binDirs.Count -eq 0) {
    throw "nvcc.exe was not found after extraction."
}

$nvcc = Join-Path $binDirs[0].FullName "nvcc.exe"
Write-Host "nvcc: $nvcc"
& $nvcc --version
