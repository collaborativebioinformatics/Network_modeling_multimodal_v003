# ============================================================
#  Synthetic Node Expansion Pipeline — Docker Image
# ============================================================
#
# CPU-only base with NetworkX / NumPy.
# On GPU hosts, swap this base for:
#   nvcr.io/nvidia/rapidsai/base:24.10-cuda12.5-py3.12
# ============================================================

FROM python:3.12-slim

LABEL maintainer="your-team"
LABEL description="Synthetic Node Expansion Pipeline"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (CPU fallback stack)
RUN pip install --no-cache-dir \
        networkx==3.3 \
        numpy==1.26.4 \
        scipy==1.13.1 \
        pyarrow==16.1.0

# Optional: install RAPIDS if building GPU image
# Uncomment for GPU variant:
# RUN pip install --no-cache-dir \
#       cupy-cuda12x \
#       cugraph-cu12 \
#       cuml-cu12

# Pipeline scripts
WORKDIR /pipeline
COPY bin/ /pipeline/bin/
RUN chmod +x /pipeline/bin/*.py

ENV PYTHONUNBUFFERED=1

CMD ["python3", "--version"]
