FROM python:3.10-slim-bookworm
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# System libs needed by OpenCV/MediaPipe
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
        git pkg-config gcc g++ python3-dev \
        libopenblas-dev liblapack-dev libatlas-base-dev \
        libjpeg-dev libpng-dev libtiff-dev \
        libavcodec-dev libavformat-dev libswscale-dev \
        libv4l-dev libxvidcore-dev libx264-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Install numpy first (helps manylinux wheel resolution)
RUN pip install numpy==1.24.3 && python -c "import numpy; print(numpy.__version__)"

# Install TensorFlow first (so protobuf/numpy get aligned)
RUN pip install tensorflow-cpu==2.13.0 && python -c "import tensorflow as tf; print(tf.__version__)"

# Install MediaPipe and let it bring its own compatible OpenCV
# Force wheel-only to avoid accidental source builds.
RUN pip install --only-binary=:all: mediapipe==0.10.0 && \
    python -c "import mediapipe as mp; print('MediaPipe', mp.__version__)"

# The rest of your stack (avoid pinning OpenCV separately)
RUN pip install \
    scipy==1.11.3 Pillow==10.0.1 scikit-learn==1.3.2 joblib==1.3.2 \
    psycopg2-binary==2.9.7 fastapi==0.104.1 uvicorn==0.22.0 python-dotenv==1.0.0

# Your smaller dependency batches (trim anything you don’t actually need)
RUN pip install \
    aiohttp==3.8.4 aiosignal==1.3.1 asgiref==3.6.0 async-timeout==4.0.2 attrs==22.2.0 \
    Babel==2.10.3 bcrypt==3.2.2 certifi==2022.9.24

RUN pip install \
    chardet==5.1.0 charset-normalizer==3.0.1 cryptography==38.0.4 defusedxml==0.7.1 \
    Django==3.2.19 frozenlist==1.3.3

RUN pip install \
    html5lib==1.1 httplib2==0.20.4 idna==3.3 importlib-metadata==4.12.0 Jinja2==3.1.2 \
    markdown-it-py==2.1.0 MarkupSafe==2.1.2 mdurl==0.1.2 more-itertools==8.10.0 \
    multidict==6.0.4 oauthlib==3.2.2 packaging==23.0 pyparsing==3.0.9 PySocks==1.7.1 \
    PyYAML==6.0 requests==2.28.1 requests-toolbelt==0.10.1 rich==13.3.1 six==1.16.0 \
    sqlparse==0.4.2 urllib3==1.26.12 webencodings==0.5.1 yarl==1.8.2 zipp==1.0.0

COPY . .
RUN mkdir -p features

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
