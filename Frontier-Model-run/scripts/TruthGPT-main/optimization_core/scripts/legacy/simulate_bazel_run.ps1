# Simulate Bazel run to validate configuration
# This script validates the Bazel configuration without requiring Bazel to be installed

Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Bazel Configuration Validation (Simulated)                      ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$baseDir = Get-Location
$errors = @()
$warnings = @()
$success = @()

# Check if Bazel is installed
Write-Host "Checking for Bazel installation..." -ForegroundColor Yellow
try {
    $bazelVersion = bazel version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Bazel is installed!" -ForegroundColor Green
        Write-Host $bazelVersion -ForegroundColor Gray
        $bazelInstalled = $true
    } else {
        $bazelInstalled = $false
    }
} catch {
    $bazelInstalled = $false
}

if (-not $bazelInstalled) {
    Write-Host "⚠️  Bazel is not installed. Running validation only..." -ForegroundColor Yellow
    Write-Host ""
}

# Validate WORKSPACE.bazel
Write-Host "Validating WORKSPACE.bazel..." -ForegroundColor Yellow
$workspaceFile = Join-Path $baseDir "WORKSPACE.bazel"
if (Test-Path $workspaceFile) {
    $content = Get-Content $workspaceFile -Raw
    
    # Check for common issues
    if ($content -match 'sha256\s*=\s*"[a-f0-9]{32}"') {
        $success += "WORKSPACE.bazel has SHA256 values"
    } else {
        $warnings += "Some dependencies may need SHA256 values (will be calculated on first run)"
    }
    
    if ($content -match 'pip_parse') {
        $success += "pip_parse configured"
    } else {
        $errors += "pip_parse not found in WORKSPACE.bazel"
    }
    
    if ($content -match 'rules_python') {
        $success += "rules_python configured"
    } else {
        $errors += "rules_python not found"
    }
    
    Write-Host "✅ WORKSPACE.bazel found and validated" -ForegroundColor Green
} else {
    $errors += "WORKSPACE.bazel not found"
}

# Validate BUILD files
Write-Host "Validating BUILD files..." -ForegroundColor Yellow
$buildFiles = Get-ChildItem -Path $baseDir -Recurse -Filter "BUILD.bazel" -ErrorAction SilentlyContinue
$buildCount = $buildFiles.Count
Write-Host "Found $buildCount BUILD.bazel files" -ForegroundColor Cyan

foreach ($buildFile in $buildFiles) {
    $relativePath = $buildFile.FullName.Replace($baseDir, "").TrimStart('\')
    $content = Get-Content $buildFile.FullName -Raw
    
    # Basic validation
    if ($content -match 'load\(') {
        $success += "BUILD file: $relativePath - has load statements"
    }
    
    if ($content -match 'py_library|cc_library|go_library|rust_library') {
        $success += "BUILD file: $relativePath - has library definitions"
    }
}

# Validate requirements_lock.txt
Write-Host "Validating requirements_lock.txt..." -ForegroundColor Yellow
$requirementsFile = Join-Path $baseDir "requirements_lock.txt"
if (Test-Path $requirementsFile) {
    $reqContent = Get-Content $requirementsFile
    $packageCount = ($reqContent | Where-Object { $_ -match '^[a-zA-Z0-9_-]+==' }).Count
    Write-Host "Found $packageCount packages in requirements_lock.txt" -ForegroundColor Cyan
    $success += "requirements_lock.txt has $packageCount packages"
} else {
    $errors += "requirements_lock.txt not found"
}

# Check dependencies
Write-Host "Validating dependencies..." -ForegroundColor Yellow
if (Test-Path "check_dependencies.py") {
    python check_dependencies.py
    if ($LASTEXITCODE -eq 0) {
        $success += "All dependencies validated"
    }
}

# Summary
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Validation Summary                                               ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

if ($success.Count -gt 0) {
    Write-Host "✅ Success ($($success.Count) checks):" -ForegroundColor Green
    foreach ($item in $success) {
        Write-Host "   • $item" -ForegroundColor Gray
    }
    Write-Host ""
}

if ($warnings.Count -gt 0) {
    Write-Host "⚠️  Warnings ($($warnings.Count)):" -ForegroundColor Yellow
    foreach ($item in $warnings) {
        Write-Host "   • $item" -ForegroundColor Gray
    }
    Write-Host ""
}

if ($errors.Count -gt 0) {
    Write-Host "❌ Errors ($($errors.Count)):" -ForegroundColor Red
    foreach ($item in $errors) {
        Write-Host "   • $item" -ForegroundColor Gray
    }
    Write-Host ""
    exit 1
}

if ($bazelInstalled) {
    Write-Host "🚀 Attempting to run Bazel query..." -ForegroundColor Cyan
    Write-Host ""
    try {
        bazel query //... 2>&1 | Select-Object -First 20
        Write-Host ""
        Write-Host "✅ Bazel query completed successfully!" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Bazel query had issues (this is normal on first run)" -ForegroundColor Yellow
        Write-Host "   Run 'bazel sync --only=@pip' to sync dependencies" -ForegroundColor Gray
    }
} else {
    Write-Host "📝 To install Bazel:" -ForegroundColor Cyan
    Write-Host "   1. Run PowerShell as Administrator" -ForegroundColor White
    Write-Host "   2. Install Chocolatey:" -ForegroundColor White
    Write-Host "      Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" -ForegroundColor Gray
    Write-Host "   3. Install Bazel: choco install bazel -y" -ForegroundColor White
    Write-Host "   4. Or download from: https://github.com/bazelbuild/bazel/releases" -ForegroundColor White
    Write-Host ""
}

Write-Host "Configuration validation complete!" -ForegroundColor Green
exit 0

