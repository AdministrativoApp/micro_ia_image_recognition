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

# Install all requirements from your file
RUN pip install -r requirements.txt

# Verify key packages
RUN python -c "import easyocr; print('EasyOCR version:', easyocr.__version__)"
RUN python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
RUN python -c "import mediapipe as mp; print('MediaPipe', mp.__version__)"

# App
COPY . .
RUN mkdir -p features

EXPOSE 443 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "443", "--ssl-keyfile", "/app/ssl/key.pem", "--ssl-certfile", "/app/ssl/cert.pem"]