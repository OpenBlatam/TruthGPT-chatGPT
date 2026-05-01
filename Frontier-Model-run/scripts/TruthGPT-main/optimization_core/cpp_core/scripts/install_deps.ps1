# PowerShell script to install C++ dependencies for optimization_core
# Run as Administrator or with appropriate permissions

param(
    [switch]$Full,
    [switch]$GPU,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Installing optimization_core C++ dependencies" -ForegroundColor Cyan
Write-Host ""

# Check for vcpkg
$vcpkgPath = $env:VCPKG_ROOT
if (-not $vcpkgPath) {
    $vcpkgPath = "C:\vcpkg"
}

if (-not (Test-Path "$vcpkgPath\vcpkg.exe")) {
    Write-Host "📦 Installing vcpkg..." -ForegroundColor Yellow
    
    git clone https://github.com/microsoft/vcpkg.git $vcpkgPath
    Push-Location $vcpkgPath
    .\bootstrap-vcpkg.bat
    Pop-Location
    
    # Add to PATH
    $env:PATH = "$vcpkgPath;$env:PATH"
    [Environment]::SetEnvironmentVariable("VCPKG_ROOT", $vcpkgPath, "User")
    [Environment]::SetEnvironmentVariable("PATH", "$vcpkgPath;$([Environment]::GetEnvironmentVariable('PATH', 'User'))", "User")
    
    Write-Host "✅ vcpkg installed" -ForegroundColor Green
} else {
    Write-Host "✅ vcpkg found at $vcpkgPath" -ForegroundColor Green
}

# Base dependencies (always installed)
$baseDeps = @(
    "eigen3:x64-windows",
    "pybind11:x64-windows"
)

# Optional dependencies
$optionalDeps = @(
    "tbb:x64-windows",
    "mimalloc:x64-windows",
    "lz4:x64-windows",
    "zstd:x64-windows"
)

# Dev dependencies
$devDeps = @(
    "gtest:x64-windows",
    "benchmark:x64-windows"
)

# GPU dependencies
$gpuDeps = @(
    # Note: CUTLASS needs to be installed separately from NVIDIA
)

Write-Host ""
Write-Host "📥 Installing base dependencies..." -ForegroundColor Yellow

foreach ($dep in $baseDeps) {
    Write-Host "  Installing $dep..." -ForegroundColor Gray
    & "$vcpkgPath\vcpkg.exe" install $dep
}

if ($Full -or $optionalDeps) {
    Write-Host ""
    Write-Host "📥 Installing optional dependencies..." -ForegroundColor Yellow
    
    foreach ($dep in $optionalDeps) {
        Write-Host "  Installing $dep..." -ForegroundColor Gray
        & "$vcpkgPath\vcpkg.exe" install $dep
    }
}

if ($Dev) {
    Write-Host ""
    Write-Host "📥 Installing dev dependencies..." -ForegroundColor Yellow
    
    foreach ($dep in $devDeps) {
        Write-Host "  Installing $dep..." -ForegroundColor Gray
        & "$vcpkgPath\vcpkg.exe" install $dep
    }
}

if ($GPU) {
    Write-Host ""
    Write-Host "⚠️ GPU dependencies require manual installation:" -ForegroundColor Yellow
    Write-Host "  1. CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    Write-Host "  2. CUTLASS: git clone https://github.com/NVIDIA/cutlass.git"
    Write-Host "  3. Set CUTLASS_ROOT environment variable"
}

Write-Host ""
Write-Host "✅ Dependencies installed!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "  1. Navigate to cpp_core directory"
Write-Host "  2. Create build directory: mkdir build && cd build"
Write-Host "  3. Configure: cmake .. -DCMAKE_TOOLCHAIN_FILE=$vcpkgPath\scripts\buildsystems\vcpkg.cmake"
Write-Host "  4. Build: cmake --build . --config Release"
Write-Host ""

# Verify installation
Write-Host "🔍 Verifying installation..." -ForegroundColor Yellow
$installed = & "$vcpkgPath\vcpkg.exe" list

foreach ($dep in $baseDeps) {
    $depName = $dep -replace ":x64-windows", ""
    if ($installed -match $depName) {
        Write-Host "  ✅ $depName" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $depName NOT FOUND" -ForegroundColor Red
    }
}












