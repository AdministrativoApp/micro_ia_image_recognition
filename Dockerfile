FROM python:3.10-slim-bookworm
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=180

# Fail fast if not amd64 (needed for MP wheels)
RUN uname -m | grep -qE 'x86_64|amd64' || (echo "ERROR: Build must target linux/amd64"; exit 1)

# System deps for TF/MP/OpenCV/EasyOCR + headers for cffi/cryptography
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    git pkg-config gcc g++ python3-dev \
    libffi-dev libssl-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy SSL certificates
COPY ssl/ /app/ssl/ 

# (Optional) print arch/Python
RUN python - <<'PY'
import platform, sys
print("ARCH:", platform.machine(), "PY:", sys.version)
PY

# Tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Fix incompatible versions before installing requirements
RUN sed -i 's/networkx==3.5/networkx==3.4.2/g' requirements.txt && \
    sed -i 's/numpy==2.2.6/numpy==1.24.3/g' requirements.txt && \
    sed -i 's/fastapi==0.116.1/fastapi==0.104.1/g' requirements.txt && \
    sed -i 's/uvicorn==0.35.0/uvicorn==0.22.0/g' requirements.txt && \
    sed -i 's/scipy==1.16.1/scipy==1.11.3/g' requirements.txt

# Install all requirements from your file
RUN pip install -r requirements.txt

# Verify key packages
RUN python -c "import easyocr; print('EasyOCR version:', easyocr.__version__)" && \
    python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)" && \
    python -c "import mediapipe as mp; print('MediaPipe', mp.__version__)" && \
    python -c "import networkx; print('NetworkX version:', networkx.__version__)"

# App
COPY . .
RUN mkdir -p features

EXPOSE 443 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "443", "--ssl-keyfile", "/app/ssl/key.pem", "--ssl-certfile", "/app/ssl/cert.pem"]