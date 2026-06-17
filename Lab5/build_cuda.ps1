$ErrorActionPreference = "Stop"

$localCuda = Join-Path $PSScriptRoot "cuda-redist"
$nvccCommand = Get-Command nvcc -ErrorAction SilentlyContinue

if ($null -eq $nvccCommand -and (Test-Path $localCuda)) {
    $nvccCommand = Get-ChildItem -Path $localCuda -Recurse -Filter nvcc.exe |
        Sort-Object FullName -Descending |
        Select-Object -First 1
}

if ($null -eq $nvccCommand) {
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaRoot) {
        $nvccCommand = Get-ChildItem -Path $cudaRoot -Recurse -Filter nvcc.exe |
            Sort-Object FullName -Descending |
            Select-Object -First 1
    }
}

if ($null -eq $nvccCommand) {
    throw "nvcc was not found. Install CUDA Toolkit or add nvcc.exe to PATH, then rerun this script."
}

$nvcc = $nvccCommand.Source
if ([string]::IsNullOrEmpty($nvcc)) {
    $nvcc = $nvccCommand.FullName
}

$nvccRoot = Split-Path (Split-Path $nvcc -Parent) -Parent
if (Test-Path $localCuda) {
    $expectedNvvm = Join-Path $nvccRoot "nvvm"
    if (!(Test-Path $expectedNvvm)) {
        $redistNvvm = Get-ChildItem -Path $localCuda -Recurse -Directory -Filter nvvm |
            Where-Object { Test-Path (Join-Path $_.FullName "bin\cicc.exe") } |
            Select-Object -First 1
        if ($null -ne $redistNvvm) {
            Copy-Item -Recurse -Force -Path $redistNvvm.FullName -Destination $expectedNvvm
        }
    }
}

$includeFlags = @()
$libFlags = @()
$pathDirs = @()

if (Test-Path $localCuda) {
    $includeDirs = Get-ChildItem -Path $localCuda -Recurse -Directory |
        Where-Object {
            (Test-Path (Join-Path $_.FullName "cuda_runtime.h")) -or
            (Test-Path (Join-Path $_.FullName "cuda/std/array")) -or
            (Test-Path (Join-Path $_.FullName "crt/host_config.h"))
        }

    foreach ($dir in $includeDirs) {
        $includeFlags += "-I$($dir.FullName)"
    }

    $cudartLib = Get-ChildItem -Path $localCuda -Recurse -Filter cudart.lib |
        Select-Object -First 1
    if ($null -ne $cudartLib) {
        $libFlags += "-L$($cudartLib.DirectoryName)"
    }

    $pathDirs = Get-ChildItem -Path $localCuda -Recurse -Directory |
        Where-Object {
            (Test-Path (Join-Path $_.FullName "nvvm64_*.dll")) -or
            (Test-Path (Join-Path $_.FullName "cudart64_*.dll")) -or
            (Test-Path (Join-Path $_.FullName "nvfatbin*.dll")) -or
            (Test-Path (Join-Path $_.FullName "nvptxcompiler*.dll"))
        } |
        ForEach-Object { $_.FullName }
}

$env:PATH = (($pathDirs + (Split-Path $nvcc -Parent) + $env:PATH) -join ";")

$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$devCmd = $null
$clDir = $null
if (Test-Path $vswhere) {
    $vsPaths = & $vswhere -all -products * -property installationPath
    foreach ($vsPath in $vsPaths) {
        if ([string]::IsNullOrWhiteSpace($vsPath)) {
            continue
        }
        $hasCl = Get-ChildItem $vsPath -Recurse -Filter cl.exe -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe" } |
            Select-Object -First 1
        $candidate = Join-Path $vsPath "Common7\Tools\LaunchDevCmd.bat"
        if (($null -ne $hasCl) -and (Test-Path $candidate)) {
            $devCmd = $candidate
            $clDir = $hasCl.DirectoryName
            break
        }
    }
}

$sourceFile = Join-Path $PSScriptRoot "main.cc"
$outputFile = Join-Path $PSScriptRoot "ann_gpu.exe"
$nvccArgs = @("-std=c++17", "-O3", "-Xcompiler", "/utf-8", "-x", "cu", $sourceFile, "-o", $outputFile) + $includeFlags + $libFlags + @("-lcudart")
if ($null -ne $clDir) {
    $nvccArgs = @("-ccbin", $clDir) + $nvccArgs
}

if ($null -ne $devCmd) {
    $quotedArgs = ($nvccArgs | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join " "
    $cmd = 'call "' + $devCmd + '" -arch=x64 -host_arch=x64 && "' + $nvcc + '" ' + $quotedArgs
    cmd.exe /c $cmd
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} else {
    & $nvcc @nvccArgs
}
