FROM python:3.11-slim-bookworm
WORKDIR /app

# System runtime deps
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        libgtk-3-0 \
        git \
        pkg-config \
        gcc \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install step by step
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.24.3