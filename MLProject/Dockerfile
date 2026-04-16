# ============================================================
# Custom Dockerfile for MLflow Model Serving
# 
# Menggunakan python:3.12-slim sebagai base image untuk menghindari
# masalah apt-get network timeout yang dialami saat mlflow models build-docker
# (error: Failed to fetch libc6, libcap2 dari Ubuntu security repository)
#
# Solusi: python:3.12-slim sudah include semua runtime yang dibutuhkan,
# sehingga hanya perlu pip install (tidak perlu wget/curl/nginx/bzip2/cmake/git)
# ============================================================
FROM python:3.12-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# ── Minimal system deps (hanya ca-certificates untuk HTTPS) ──
# Konfigurasi apt retry untuk menghindari network timeout
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "60";' >> /etc/apt/apt.conf.d/80-retries && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────
COPY requirements-serve.txt /tmp/requirements-serve.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements-serve.txt

# ── Copy model artifacts ─────────────────────────────────────
WORKDIR /app
COPY model/ /app/model/

# ── Expose port ──────────────────────────────────────────────
EXPOSE 8080

# ── Healthcheck ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# ── Entrypoint ───────────────────────────────────────────────
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "8080", "--no-conda"]
