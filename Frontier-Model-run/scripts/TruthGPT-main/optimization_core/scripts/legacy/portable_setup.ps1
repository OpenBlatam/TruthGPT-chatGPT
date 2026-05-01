# Check if running with administrative privileges (optional but helpful for some setups)
# $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
# if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
#     Write-Warning "Not running as Administrator. Some local path registrations might require manual PATH updates."
# }

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   OPENCLAW - PORTABLE SETUP" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$PythonCmd = "python"
try {
    & $PythonCmd --version > $null 2>&1
} catch {
    Write-Error "Python not found. Please install Python 3.10+ and add it to your PATH."
    exit 1
}

Write-Host "➤ Detecting environment..." -ForegroundColor Gray
$InstallDir = $PSScriptRoot

Write-Host "➤ Installing OpenClaw in editable mode..." -ForegroundColor Cyan
# Using 'python -m pip' is more robust than 'pip' directly
& $PythonCmd -m pip install -e $InstallDir

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Installation Successful!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "You can now use the 'openclaw' command directly."
    Write-Host "Try running:" -ForegroundColor Gray
    Write-Host "   openclaw --help" -ForegroundColor Yellow
    Write-Host "   openclaw tools --list" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Cyan
} else {
    Write-Host "❌ Installation failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
