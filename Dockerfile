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

# Install packages in small groups
RUN head -20 requirements.txt > requirements-part1.txt && \
    pip install --no-cache-dir -r requirements-part1.txt

RUN tail -n +21 requirements.txt | head -20 > requirements-part2.txt && \
    pip install --no-cache-dir -r requirements-part2.txt

RUN tail -n +41 requirements.txt > requirements-part3.txt && \
    pip install --no-cache-dir -r requirements-part3.txt

# Verify core packages
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    opencv-python-headless==4.8.1.78 \
    fastapi==0.104.1 \
    uvicorn==0.22.0 \
    Pillow==10.0.1

COPY . .
RUN mkdir -p features

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]