$ErrorActionPreference = "Stop"

$installer = Join-Path $PSScriptRoot "vs_BuildTools.exe"
$url = "https://aka.ms/vs/17/release/vs_BuildTools.exe"

if (!(Test-Path $installer)) {
    Write-Host "Downloading Visual Studio Build Tools bootstrapper"
    curl.exe -L $url -o $installer
}

Write-Host "Installing MSVC C++ Build Tools. This can take several minutes."
& $installer --quiet --wait --norestart --nocache `
    --add Microsoft.VisualStudio.Workload.VCTools `
    --includeRecommended

$cl = @(
    Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter cl.exe -ErrorAction SilentlyContinue
    Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio" -Recurse -Filter cl.exe -ErrorAction SilentlyContinue
) |
    Where-Object { $_.FullName -like "*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe" } |
    Sort-Object FullName -Descending |
    Select-Object -First 1

if ($null -eq $cl) {
    throw "MSVC cl.exe was not found after installation."
}

Write-Host "cl: $($cl.FullName)"
& $cl.FullName
