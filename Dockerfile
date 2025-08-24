FROM python:3.11-slim-bookworm

WORKDIR /app

# System deps — only runtime libs (no *-dev needed for pip wheels)
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        libgtk-3-0 \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# If you use OpenCV, prefer the headless wheel in servers/containers
# (no GUI backends; avoids extra deps). In requirements.txt, use:
# opencv-python-headless>=4.9,<5
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p features

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
