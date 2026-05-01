# ════════════════════════════════════════════════════════════════════════════════
# TruthGPT Go Core - Dependency Installation Script (Windows)
# ════════════════════════════════════════════════════════════════════════════════

Write-Host "🐹 TruthGPT Go Core - Installing Dependencies" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Check Go installation
try {
    $goVersion = (go version) -replace "go version go", "" -replace " .*", ""
    Write-Host "✅ Go version: $goVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Go is not installed. Please install Go from https://go.dev/dl/" -ForegroundColor Red
    exit 1
}

# Navigate to go_core directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$scriptDir\.."

Write-Host ""
Write-Host "📦 Downloading Go modules..." -ForegroundColor Yellow
go mod download

Write-Host ""
Write-Host "🔧 Verifying modules..." -ForegroundColor Yellow
go mod verify

Write-Host ""
Write-Host "🔨 Building binaries..." -ForegroundColor Yellow
go build ./...

Write-Host ""
Write-Host "🧪 Running tests..." -ForegroundColor Yellow
go test ./... -v -short

Write-Host ""
Write-Host "📊 Installing additional tools..." -ForegroundColor Yellow

# Install protoc-gen-go
Write-Host "   Installing protoc-gen-go..." -ForegroundColor Gray
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest

# Install protoc-gen-go-grpc  
Write-Host "   Installing protoc-gen-go-grpc..." -ForegroundColor Gray
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

Write-Host ""
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run 'make build' to build all binaries"
Write-Host "  2. Run 'make test' to run tests"
Write-Host "  3. Run 'make run-inference' to start the inference server"
Write-Host ""












