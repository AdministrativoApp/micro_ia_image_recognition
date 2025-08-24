FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for OpenCV and image processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get install -y --no-install-recommends \
    ffmpeg \
    && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && apt-get install -y --no-install-recommends \
    libgtk-3-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Create required folder
RUN mkdir -p features

# Set environment
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
