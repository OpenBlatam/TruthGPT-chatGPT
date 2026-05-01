# Simple Bazel validation script
Write-Host "Bazel Configuration Validation" -ForegroundColor Cyan
Write-Host ""

# Check if Bazel is installed
$bazelInstalled = $false
try {
    $null = bazel version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Bazel is installed!" -ForegroundColor Green
        bazel version
        $bazelInstalled = $true
    }
} catch {
    $bazelInstalled = $false
}

if (-not $bazelInstalled) {
    Write-Host "Bazel is not installed." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To install Bazel:" -ForegroundColor Cyan
    Write-Host "1. Run PowerShell as Administrator" -ForegroundColor White
    Write-Host "2. Install Chocolatey, then: choco install bazel -y" -ForegroundColor White
    Write-Host "3. Or download from: https://github.com/bazelbuild/bazel/releases" -ForegroundColor White
    Write-Host ""
}

# Validate files exist
Write-Host "Validating configuration files..." -ForegroundColor Yellow
$files = @("WORKSPACE.bazel", "BUILD.bazel", "requirements_lock.txt")
foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "  Found: $file" -ForegroundColor Green
    } else {
        Write-Host "  Missing: $file" -ForegroundColor Red
    }
}

# Count BUILD files
$buildFiles = Get-ChildItem -Recurse -Filter "BUILD.bazel" -ErrorAction SilentlyContinue
Write-Host "  Found $($buildFiles.Count) BUILD.bazel files" -ForegroundColor Green

# Run dependency check
if (Test-Path "check_dependencies.py") {
    Write-Host ""
    Write-Host "Checking dependencies..." -ForegroundColor Yellow
    python check_dependencies.py
}

# If Bazel is installed, try to run query
if ($bazelInstalled) {
    Write-Host ""
    Write-Host "Running Bazel query..." -ForegroundColor Cyan
    bazel query //... 2>&1 | Select-Object -First 30
}

Write-Host ""
Write-Host "Validation complete!" -ForegroundColor Green












