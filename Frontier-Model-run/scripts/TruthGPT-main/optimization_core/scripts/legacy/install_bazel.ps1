# Install Bazel on Windows
# This script provides multiple installation methods

Write-Host "Installing Bazel on Windows..." -ForegroundColor Green

# Method 1: Check if Chocolatey is available
$chocoAvailable = $false
try {
    $chocoVersion = choco --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        $chocoAvailable = $true
        Write-Host "Chocolatey found: $chocoVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "Chocolatey not found" -ForegroundColor Yellow
}

if ($chocoAvailable) {
    Write-Host "Installing Bazel via Chocolatey..." -ForegroundColor Cyan
    choco install bazel -y
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Bazel installed successfully!" -ForegroundColor Green
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        exit 0
    }
}

# Method 2: Download and install manually
Write-Host "`nChocolatey not available. Manual installation required." -ForegroundColor Yellow
Write-Host "`nPlease install Bazel using one of these methods:" -ForegroundColor Cyan
Write-Host "1. Install Chocolatey first:" -ForegroundColor White
Write-Host "   Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" -ForegroundColor Gray
Write-Host "`n2. Then run: choco install bazel -y" -ForegroundColor White
Write-Host "`n3. Or download manually from:" -ForegroundColor White
Write-Host "   https://github.com/bazelbuild/bazel/releases" -ForegroundColor Gray
Write-Host "`n4. Or use Bazelisk (Bazel version manager):" -ForegroundColor White
Write-Host "   choco install bazelisk -y" -ForegroundColor Gray

Write-Host "`nAfter installation, verify with: bazel version" -ForegroundColor Cyan












