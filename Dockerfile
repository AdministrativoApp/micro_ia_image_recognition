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

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Verify ALL critical packages
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
import mediapipe; print('MediaPipe OK'); \
from dotenv import load_dotenv; print('python-dotenv OK'); \
print('All imports successful!')\
"

COPY . .
RUN mkdir -p features

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]