# 🛠️ TruthGPT SOTA 2025 - Global Installer
# This script installs TruthGPT and enables the 'truth' command globally.

$ErrorActionPreference = "Stop"

function Write-Header($text) {
    Write-Host "`n=== $text ===" -ForegroundColor Cyan
}

function Write-Success($text) {
    Write-Host "✓ $text" -ForegroundColor Green
}

function Write-Error-Custom($text) {
    Write-Host "✗ $text" -ForegroundColor Red
}

Write-Header "Starting TruthGPT Industrial Installation"

# 1. Check Python
Write-Host "Checking environment..."
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Python found: $pythonVersion"
} catch {
    Write-Error-Custom "Python not found. Please install Python 3.10 or higher."
    exit 1
}

# 2. Define Installation Directory
$installDir = Join-Path $env:USERPROFILE ".truthgpt"
if (-not (Test-Path $installDir)) {
    New-Item -Path $installDir -ItemType Directory | Out-Null
    Write-Success "Created installation directory: $installDir"
}

# 3. Download / Sync Codebase
Write-Header "Downloading TruthGPT Core"
$repoUrl = "https://github.com/OpenBlatam/IA-Models-Clone.git"

if (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "Git detected. Cloning repository..."
    if (Test-Path (Join-Path $installDir ".git")) {
        Push-Location $installDir
        git pull
        Pop-Location
    } else {
        git clone $repoUrl $installDir
    }
} else {
    Write-Host "Git not found. Downloading ZIP..."
    $zipPath = Join-Path $env:TEMP "truthgpt.zip"
    # Note: ZIP URL for GitHub is /archive/refs/heads/main.zip
    $zipUrl = $repoUrl.Replace(".git", "") + "/archive/refs/heads/main.zip"
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $installDir -Force
}

# 4. Install Dependencies
Write-Header "Installing Dependencies"
Push-Location $installDir

# Detect where pyproject.toml is (it might be in a subdirectory after unzipping)
$projectFile = Get-ChildItem -Filter "pyproject.toml" -Recurse | Select-Object -First 1
if ($projectFile) {
    $projectDir = $projectFile.DirectoryName
    Write-Host "Found project in: $projectDir"
    Push-Location $projectDir
} else {
    Write-Error-Custom "Could not find pyproject.toml in the downloaded files."
    exit 1
}

Write-Host "Running pip install (this may take a few minutes)..."
python -m pip install --upgrade pip
python -m pip install -e .

# 5. Verify 'truth' command
Write-Header "Verification"
$scriptsPath = python -c "import sysconfig; print(sysconfig.get_path('scripts'))"
Write-Host "Python Scripts directory: $scriptsPath"

# Add to User PATH if not present
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$scriptsPath*") {
    Write-Host "Adding $scriptsPath to User PATH..."
    [Environment]::SetEnvironmentVariable("Path", $currentPath + ";" + $scriptsPath, "User")
    # Also update current session
    $env:Path += ";" + $scriptsPath
    Write-Success "PATH updated permanently for the user."
} else {
    Write-Success "Scripts directory already in PATH."
}

# Final check
if (Get-Command truth -ErrorAction SilentlyContinue) {
    Write-Success "The 'truth' command is now available!"
} else {
    Write-Warning "The 'truth' command was installed but is not yet in the current session's PATH. Please restart your terminal."
}

Pop-Location # Back from $projectDir
Pop-Location # Back from $installDir

Write-Header "Installation Complete"
Write-Host "You can now use the 'truth' command from any terminal." -ForegroundColor Green
Write-Host "Try: truth --help"
