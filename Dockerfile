FROM python:3.11-slim

WORKDIR /app

# Update package lists and install essential packages in separate steps
RUN apt-get update

# Install core dependencies first
RUN apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    wget \
    curl

# Install OpenCV dependencies
RUN apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

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
