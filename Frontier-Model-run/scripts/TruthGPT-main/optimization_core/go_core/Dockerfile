# ════════════════════════════════════════════════════════════════════════════════
# TruthGPT Go Core - Multi-stage Dockerfile
# ════════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────────
# Stage 1: Build
# ────────────────────────────────────────────────────────────────────────────────
FROM golang:1.22-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /build

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build all binaries with optimizations
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-s -w -X main.Version=1.0.0 -X main.BuildTime=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    -o /bin/inference-server ./cmd/inference-server

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-s -w -X main.Version=1.0.0 -X main.BuildTime=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    -o /bin/cache-service ./cmd/cache-service

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-s -w -X main.Version=1.0.0 -X main.BuildTime=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    -o /bin/coordinator ./cmd/coordinator

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-s -w -X main.Version=1.0.0 -X main.BuildTime=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    -o /bin/data-pipeline ./cmd/data-pipeline

# ────────────────────────────────────────────────────────────────────────────────
# Stage 2: Inference Server Runtime
# ────────────────────────────────────────────────────────────────────────────────
FROM scratch AS inference-server

# Copy CA certificates for HTTPS
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy timezone data
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Copy binary
COPY --from=builder /bin/inference-server /inference-server

# Expose ports
EXPOSE 8080 50051 9090

# Run
ENTRYPOINT ["/inference-server"]

# ────────────────────────────────────────────────────────────────────────────────
# Stage 3: Cache Service Runtime
# ────────────────────────────────────────────────────────────────────────────────
FROM alpine:3.19 AS cache-service

# Install minimal dependencies
RUN apk add --no-cache ca-certificates tzdata

# Create non-root user
RUN adduser -D -H -s /sbin/nologin truthgpt
USER truthgpt

# Copy binary
COPY --from=builder /bin/cache-service /cache-service

# Create data directory
VOLUME ["/data"]

# Expose port
EXPOSE 8081

# Run
ENTRYPOINT ["/cache-service", "--badger-path=/data/kv_cache"]

# ────────────────────────────────────────────────────────────────────────────────
# Stage 4: All-in-one Runtime
# ────────────────────────────────────────────────────────────────────────────────
FROM alpine:3.19 AS all-in-one

# Install minimal dependencies
RUN apk add --no-cache ca-certificates tzdata supervisor

# Create non-root user
RUN adduser -D -H -s /sbin/nologin truthgpt

# Copy all binaries
COPY --from=builder /bin/inference-server /usr/local/bin/
COPY --from=builder /bin/cache-service /usr/local/bin/
COPY --from=builder /bin/coordinator /usr/local/bin/
COPY --from=builder /bin/data-pipeline /usr/local/bin/

# Create supervisor config
RUN mkdir -p /etc/supervisor.d
COPY <<EOF /etc/supervisor.d/truthgpt.ini
[supervisord]
nodaemon=true
user=root

[program:inference-server]
command=/usr/local/bin/inference-server
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:cache-service]
command=/usr/local/bin/cache-service --badger-path=/data/kv_cache
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
EOF

# Create data directory
VOLUME ["/data"]

# Expose ports
EXPOSE 8080 8081 50051 9090

# Run supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]

# ────────────────────────────────────────────────────────────────────────────────
# Default target
# ────────────────────────────────────────────────────────────────────────────────
FROM inference-server












