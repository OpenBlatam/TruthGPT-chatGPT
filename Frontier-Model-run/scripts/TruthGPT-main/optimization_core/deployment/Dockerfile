# Ultra Speed TruthGPT Dockerfile for AWS EKS
# Multi-stage build for optimal performance and security

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    vim \
    htop \
    tree \
    jq \
    unzip \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r truthgpt && useradd -r -g truthgpt truthgpt

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development dependencies
FROM base as development

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit \
    jupyter \
    ipython

# Stage 3: Production dependencies
FROM base as production

# Install production dependencies
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn \
    fastapi \
    pydantic \
    redis \
    celery \
    flower \
    prometheus-client \
    structlog \
    sentry-sdk

# Stage 4: GPU support
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    wget \
    vim \
    htop \
    tree \
    jq \
    unzip \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Create non-root user
RUN groupadd -r truthgpt && useradd -r -g truthgpt truthgpt

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with GPU support
RUN pip install --no-cache-dir -r requirements.txt

# Install additional GPU dependencies
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Stage 5: Final production image
FROM production as final

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/cache /app/logs /app/config /app/plugins /tmp && \
    chown -R truthgpt:truthgpt /app

# Set permissions
RUN chmod +x /app/scripts/*.sh

# Switch to non-root user
USER truthgpt

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Default command (Using the modular CLI)
CMD ["python", "cli.py", "swarm", "ask", "Hola, ¿en qué puedes ayudarme hoy?"]

# Labels for metadata
LABEL maintainer="TruthGPT Team" \
      version="1.0.0" \
      description="Ultra Speed TruthGPT Optimization System" \
      optimization="ultra-speed" \
      gpu="enabled" \
      distributed="enabled" \
      quantization="enabled" \
      monitoring="enabled" \
      logging="enabled" \
      security="enabled"

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Metadata labels
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0" \
      org.label-schema.name="ultra-speed-truthgpt" \
      org.label-schema.description="Ultra Speed TruthGPT Optimization System" \
      org.label-schema.vendor="TruthGPT" \
      org.label-schema.url="https://github.com/truthgpt/ultra-speed-optimization" \
      org.label-schema.vcs-url="https://github.com/truthgpt/ultra-speed-optimization.git" \
      org.label-schema.docker.cmd="docker run -d --name ultra-speed-truthgpt -p 8080:8080 -p 9090:9090 ultra-speed-truthgpt:latest"









