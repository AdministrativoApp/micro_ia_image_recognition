FROM python:3.11-slim-bookworm
WORKDIR /app

# Only essential system dependencies for web/imaging
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # For OpenCV/image processing
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        
        # Development/build tools
        git \
        pkg-config \
        gcc \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Clean requirements.txt - remove all desktop/GUI packages
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Verify critical web packages
RUN python -c "import fastapi; print('FastAPI version:', fastapi.__version__)" && \
    python -c "import uvicorn; print('Uvicorn version:', uvicorn.__version__)" && \
    python -c "import numpy; print('NumPy version:', numpy.__version__)" && \
    python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# App files
COPY . .
RUN mkdir -p features

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]