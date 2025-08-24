FROM python:3.11-bullseye

WORKDIR /app

# Install system dependencies needed for OpenCV and image processing
RUN echo "-- apt sources:" && cat /etc/apt/sources.list || true && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgtk-3-0 \
        pkg-config && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

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
