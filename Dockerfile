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

# Then install computer vision packages
RUN pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78 \
    scikit-learn==1.3.2 \
    mediapipe==0.10.0

# Then install tensorflow (can be problematic)
RUN pip install --no-cache-dir tensorflow==2.13.0

# Then install database and web packages
RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.7 \
    fastapi==0.104.1 \
    uvicorn==0.22.0

# Finally install the rest from requirements.txt in batches
RUN head -20 requirements.txt > requirements-part1.txt && \
    pip install --no-cache-dir -r requirements-part1.txt

RUN tail -n +21 requirements.txt | head -20 > requirements-part2.txt && \
    pip install --no-cache-dir -r requirements-part2.txt

RUN tail -n +41 requirements.txt > requirements-part3.txt && \
    pip install --no-cache-dir -r requirements-part3.txt

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