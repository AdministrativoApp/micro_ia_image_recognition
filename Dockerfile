FROM python:3.11-slim-bookworm
WORKDIR /app

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

COPY requirements.txt .

RUN pip install --upgrade pip

# Install foundational packages first
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    Pillow==10.0.1 \
    joblib==1.3.2 \
    python-dotenv==1.0.0

# Install OpenCV
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78

# Install scikit-learn
RUN pip install --no-cache-dir scikit-learn==1.3.2

# Install tensorflow-cpu instead of tensorflow
RUN pip install --no-cache-dir tensorflow-cpu==2.13.0

# Install database and web packages
RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.7 \
    fastapi==0.104.1 \
    uvicorn==0.22.0

# Create a simple filtered requirements file using grep
RUN grep -v -E "(numpy|Pillow|joblib|python-dotenv|opencv-python-headless|scikit-learn|tensorflow|psycopg2-binary|fastapi|uvicorn|mediapipe)" requirements.txt > filtered-requirements.txt

# Install the remaining packages
RUN pip install --no-cache-dir -r filtered-requirements.txt

# Verify ALL critical packages (without mediapipe)
RUN python -c "\
import fastapi; print('FastAPI OK'); \
import uvicorn; print('Uvicorn OK'); \
import numpy; print('NumPy OK'); \
import cv2; print('OpenCV OK'); \
import PIL; print('Pillow OK'); \
import joblib; print('Joblib OK'); \
import psycopg2; print('Psycopg2 OK'); \
import sklearn; print('Scikit-learn OK'); \
import tensorflow; print('TensorFlow OK'); \
from dotenv import load_dotenv; print('python-dotenv OK'); \
print('All imports successful!')\
"

COPY . .
RUN mkdir -p features

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]