# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# Telecom RAG — production image (multi-stage)
# -----------------------------------------------------------------------------
# Stage 1 (builder): full toolchain, builds the Python venv.
# Stage 2 (runtime): slim base + the venv from stage 1; no compiler.
# Net win: the ~336 MB build-essential layer does NOT land in the final
# image. Expected image size: ~1.2-1.4 GB (down from the ~2.0 GB
# single-stage build). AC10 cap is 1.5 GB; if multi-stage alone isn't
# enough, the dep-set audit is the follow-up path.
# -----------------------------------------------------------------------------
# Layer order matters for cache reuse:
#   1. System deps (rarely change)
#   2. requirements.txt COPY (changes only when deps change)
#   3. pip install (rebuilds only when requirements.txt changes)
#   4. App code COPY (changes every commit)
# Splitting (2)+(3) from (4) keeps a normal code change from invalidating
# the slow pip install layer.
# -----------------------------------------------------------------------------

# ===========================================================================
# Stage 1 — builder
# ===========================================================================
# Full toolchain: needed only to compile some wheels (numpy/scipy/chromadb's
# transitive stack) on first install. The runtime image does not need it.
FROM python:3.11-slim AS builder

# Don't write .pyc files, don't buffer stdout (matters for log shipping),
# fail fast on pip issues.  Telemetry-related envs (LangSmith) must be set
# at runtime via .env.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=random

# ---------- System dependencies (build stage only) ----------
# build-essential compiles wheels for some packages on first install.
# We strip the apt cache in the same RUN to keep the layer small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Python dependencies ----------
# Install into a PEP 405 venv at /opt/venv so the runtime stage can copy
# it as a single layer without re-running pip.  /opt/venv is the
# conventional location.
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt \
    && /opt/venv/bin/pip check

# ===========================================================================
# Stage 2 — runtime
# ===========================================================================
# Slim base; carries the venv from the builder, but NOT the compiler.
FROM python:3.11-slim AS runtime

# Same runtime envs as the builder so behavior is consistent.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=random

# ---------- Runtime system dependencies ----------
# curl + ca-certificates: used by the HEALTHCHECK below.
# No build-essential here — that's the point of the multi-stage split.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the venv from the builder stage.  This is the single biggest win:
# the pip install layer (and its compile-time footprint) lands as a
# read-only copy in the runtime image, not as a fresh install.
COPY --from=builder /opt/venv /opt/venv

# Make the venv the default Python for subsequent RUN/CMD.
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code last so day-to-day code changes don't bust the
# slow pip layer above.
COPY . .

# ---------- Runtime user ----------
# Default to a non-root user.  UID 10001 is high enough to not collide
# with host users on a typical Linux host.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --shell /bin/bash --create-home app \
    && chown -R app:app /app
USER app

# Streamlit default port.  Override with `docker run -p ...` as needed.
EXPOSE 8501

# Smoke test on every container start.  This catches a broken install
# (e.g. a wheel that fails to import) before it silently serves traffic.
# Kept tiny on purpose — real readiness probes belong in compose/k8s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "from telecom_rag.config import settings; print(settings.llm_model)" \
        || exit 1

# Default command.  Today this prints the configured LLM model as a
# "the image is wired up correctly" signal.  When a Streamlit entrypoint
# lands (Issue #5), this becomes:
#   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["python", "-c", "from telecom_rag.config import settings; print('telecom-rag ready, model=', settings.llm_model)"]
