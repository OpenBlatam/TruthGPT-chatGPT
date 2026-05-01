"""
🐳 Dockerfile for Production Inference API
Multi-stage build optimized for production
"""

FROM python:3.11-slim as builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements_advanced.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_advanced.txt || \
    pip install --no-cache-dir fastapi uvicorn httpx pydantic redis psutil || true

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser inference/ ./inference/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser configs/llm_default.yaml ./configs/llm_default.yaml

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE ${PORT}

# Run application
CMD ["python", "-m", "uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]


