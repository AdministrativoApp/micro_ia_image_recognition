FROM python:3.11-slim-bookworm
WORKDIR /app

# System dependencies
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        git \
        pkg-config \
        gcc \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Verify installations
RUN python -c "import fastapi; print('FastAPI OK')" && \
    python -c "import uvicorn; print('Uvicorn OK')" && \
    python -c "import numpy; print('NumPy OK')" && \
    python -c "import cv2; print('OpenCV OK')" && \
    python -c "import PIL; print('Pillow OK')"

# App files
COPY . .
RUN mkdir -p features

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]