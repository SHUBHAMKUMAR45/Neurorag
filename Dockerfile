# ─── NeuroRAG Production Dockerfile ──────────────────────────────────────────
# Multi-stage build: slim final image with CUDA support for RTX 4060.
# Base: PyTorch CUDA runtime (CUDA 12.1, cuDNN 8)

# ── Stage 1: Builder (compile deps) ──────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS builder

WORKDIR /build

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
 && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS runtime

LABEL org.opencontainers.image.title="NeuroRAG"
LABEL org.opencontainers.image.version="3.0.0"
LABEL org.opencontainers.image.description="Autonomous Self-Healing Multi-Agent RAG System"

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    tini \
 && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /opt/conda /opt/conda

# Copy application source
COPY . .

# Create non-root user
RUN useradd -m -u 1000 neurorag \
 && mkdir -p /data/faiss /data/whoosh_index /data/drift /models /logs \
 && chown -R neurorag:neurorag /app /data /logs

USER neurorag

# Expose API + Prometheus ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--log-level", "info"]
