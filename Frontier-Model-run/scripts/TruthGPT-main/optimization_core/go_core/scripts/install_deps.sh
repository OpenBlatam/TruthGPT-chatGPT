#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════════
# TruthGPT Go Core - Dependency Installation Script (Linux/macOS)
# ════════════════════════════════════════════════════════════════════════════════

set -e

echo "🐹 TruthGPT Go Core - Installing Dependencies"
echo "=============================================="

# Check Go version
GO_VERSION=$(go version 2>/dev/null | grep -oP 'go\K[0-9]+\.[0-9]+' || echo "0.0")
REQUIRED_VERSION="1.22"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$GO_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "❌ Go $REQUIRED_VERSION or higher is required (found: $GO_VERSION)"
    echo "   Please install Go from https://go.dev/dl/"
    exit 1
fi

echo "✅ Go version: $GO_VERSION"

# Navigate to go_core directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo ""
echo "📦 Downloading Go modules..."
go mod download

echo ""
echo "🔧 Verifying modules..."
go mod verify

echo ""
echo "🔨 Building binaries..."
go build ./...

echo ""
echo "🧪 Running tests..."
go test ./... -v -short

echo ""
echo "📊 Installing additional tools..."

# Install golangci-lint
if ! command -v golangci-lint &> /dev/null; then
    echo "   Installing golangci-lint..."
    curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin v1.55.2
fi

# Install protoc-gen-go
if ! command -v protoc-gen-go &> /dev/null; then
    echo "   Installing protoc-gen-go..."
    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
fi

# Install protoc-gen-go-grpc
if ! command -v protoc-gen-go-grpc &> /dev/null; then
    echo "   Installing protoc-gen-go-grpc..."
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Run 'make build' to build all binaries"
echo "  2. Run 'make test' to run tests"
echo "  3. Run 'make run-inference' to start the inference server"
echo ""












