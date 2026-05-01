# ════════════════════════════════════════════════════════════════════════════════
# TruthGPT Go Core - Makefile
# ════════════════════════════════════════════════════════════════════════════════

.PHONY: all build test bench lint proto docker clean help

# Variables
GO := go
GOFLAGS := -ldflags="-s -w"
BINDIR := bin
PROTODIR := proto
COVERAGE := coverage.out

# Binaries
INFERENCE_SERVER := $(BINDIR)/inference-server
CACHE_SERVICE := $(BINDIR)/cache-service
COORDINATOR := $(BINDIR)/coordinator
DATA_PIPELINE := $(BINDIR)/data-pipeline

# ════════════════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ════════════════════════════════════════════════════════════════════════════════

all: build

# ════════════════════════════════════════════════════════════════════════════════
# BUILD TARGETS
# ════════════════════════════════════════════════════════════════════════════════

build: $(INFERENCE_SERVER) $(CACHE_SERVICE) $(COORDINATOR) $(DATA_PIPELINE)

$(BINDIR):
	mkdir -p $(BINDIR)

$(INFERENCE_SERVER): $(BINDIR)
	$(GO) build $(GOFLAGS) -o $@ ./cmd/inference-server

$(CACHE_SERVICE): $(BINDIR)
	$(GO) build $(GOFLAGS) -o $@ ./cmd/cache-service

$(COORDINATOR): $(BINDIR)
	$(GO) build $(GOFLAGS) -o $@ ./cmd/coordinator

$(DATA_PIPELINE): $(BINDIR)
	$(GO) build $(GOFLAGS) -o $@ ./cmd/data-pipeline

build-all: build
	@echo "✅ All binaries built successfully"

# ════════════════════════════════════════════════════════════════════════════════
# TEST TARGETS
# ════════════════════════════════════════════════════════════════════════════════

test:
	$(GO) test ./... -v -race

test-short:
	$(GO) test ./... -v -short

test-coverage:
	$(GO) test ./... -coverprofile=$(COVERAGE) -covermode=atomic
	$(GO) tool cover -html=$(COVERAGE) -o coverage.html
	@echo "✅ Coverage report: coverage.html"

# ════════════════════════════════════════════════════════════════════════════════
# BENCHMARK TARGETS
# ════════════════════════════════════════════════════════════════════════════════

bench:
	$(GO) test ./benchmarks/... -bench=. -benchmem -run=^$

bench-cache:
	$(GO) test ./internal/cache/... -bench=. -benchmem -run=^$

bench-compression:
	$(GO) test ./internal/compression/... -bench=. -benchmem -run=^$

# ════════════════════════════════════════════════════════════════════════════════
# LINT & FORMAT
# ════════════════════════════════════════════════════════════════════════════════

lint:
	golangci-lint run ./...

fmt:
	$(GO) fmt ./...
	goimports -w .

vet:
	$(GO) vet ./...

# ════════════════════════════════════════════════════════════════════════════════
# PROTOBUF
# ════════════════════════════════════════════════════════════════════════════════

proto:
	protoc --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		$(PROTODIR)/*.proto
	@echo "✅ Protobuf files generated"

# ════════════════════════════════════════════════════════════════════════════════
# DOCKER
# ════════════════════════════════════════════════════════════════════════════════

docker:
	docker build -t truthgpt-go-core:latest .

docker-run:
	docker run -p 8080:8080 -p 50051:50051 -p 9090:9090 truthgpt-go-core:latest

# ════════════════════════════════════════════════════════════════════════════════
# RUN TARGETS
# ════════════════════════════════════════════════════════════════════════════════

run-inference:
	$(GO) run ./cmd/inference-server

run-cache:
	$(GO) run ./cmd/cache-service

run-coordinator:
	$(GO) run ./cmd/coordinator

run-pipeline:
	$(GO) run ./cmd/data-pipeline

# ════════════════════════════════════════════════════════════════════════════════
# DEPENDENCIES
# ════════════════════════════════════════════════════════════════════════════════

deps:
	$(GO) mod download
	$(GO) mod verify

deps-update:
	$(GO) get -u ./...
	$(GO) mod tidy

deps-tidy:
	$(GO) mod tidy

# ════════════════════════════════════════════════════════════════════════════════
# CLEAN
# ════════════════════════════════════════════════════════════════════════════════

clean:
	rm -rf $(BINDIR)
	rm -f $(COVERAGE) coverage.html
	$(GO) clean -cache -testcache

# ════════════════════════════════════════════════════════════════════════════════
# HELP
# ════════════════════════════════════════════════════════════════════════════════

help:
	@echo "TruthGPT Go Core - Available Commands"
	@echo ""
	@echo "Build:"
	@echo "  make build              Build all binaries"
	@echo "  make build-all          Build all binaries with verification"
	@echo ""
	@echo "Test:"
	@echo "  make test               Run all tests with race detection"
	@echo "  make test-short         Run short tests only"
	@echo "  make test-coverage      Run tests with coverage report"
	@echo ""
	@echo "Benchmark:"
	@echo "  make bench              Run all benchmarks"
	@echo "  make bench-cache        Run cache benchmarks"
	@echo "  make bench-compression  Run compression benchmarks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint               Run golangci-lint"
	@echo "  make fmt                Format code"
	@echo "  make vet                Run go vet"
	@echo ""
	@echo "Run:"
	@echo "  make run-inference      Run inference server"
	@echo "  make run-cache          Run cache service"
	@echo "  make run-coordinator    Run coordinator"
	@echo "  make run-pipeline       Run data pipeline"
	@echo ""
	@echo "Docker:"
	@echo "  make docker             Build Docker image"
	@echo "  make docker-run         Run Docker container"
	@echo ""
	@echo "Dependencies:"
	@echo "  make deps               Download dependencies"
	@echo "  make deps-update        Update all dependencies"
	@echo "  make deps-tidy          Tidy go.mod"
	@echo ""
	@echo "Other:"
	@echo "  make proto              Generate protobuf code"
	@echo "  make clean              Clean build artifacts"
	@echo "  make help               Show this help"












