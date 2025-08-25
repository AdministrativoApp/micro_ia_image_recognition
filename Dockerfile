FROM python:3.11-slim-bookworm
WORKDIR /app

# Install ALL required system dependencies for scientific packages
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        git \
        pkg-config \
        gcc \
        g++ \
        python3-dev \
        libopenblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install and verify each critical package one by one
RUN pip install --no-cache-dir numpy==1.24.3 && \
    python -c "import numpy; print('NumPy OK - version:', numpy.__version__)"

RUN pip install --no-cache-dir scipy==1.11.3 && \
    python -c "import scipy; print('SciPy OK - version:', scipy.__version__)"

RUN pip install --no-cache-dir Pillow==10.0.1 && \
    python -c "import PIL; print('Pillow OK - version:', PIL.__version__)"

RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78 && \
    python -c "import cv2; print('OpenCV OK - version:', cv2.__version__)"

# Install scikit-learn with build dependencies
RUN pip install --no-cache-dir scikit-learn==1.3.2 && \
    python -c "import sklearn; print('Scikit-learn OK - version:', sklearn.__version__)"

RUN pip install --no-cache-dir joblib==1.3.2 && \
    python -c "import joblib; print('Joblib OK - version:', joblib.__version__)"

RUN pip install --no-cache-dir tensorflow-cpu==2.13.0 && \
    python -c "import tensorflow; print('TensorFlow OK - version:', tensorflow.__version__)"

RUN pip install --no-cache-dir psycopg2-binary==2.9.7 && \
    python -c "import psycopg2; print('Psycopg2 OK - version:', psycopg2.__version__)"

RUN pip install --no-cache-dir fastapi==0.104.1 && \
    python -c "import fastapi; print('FastAPI OK - version:', fastapi.__version__)"

RUN pip install --no-cache-dir uvicorn==0.22.0 && \
    python -c "import uvicorn; print('Uvicorn OK - version:', uvicorn.__version__)"

RUN pip install --no-cache-dir python-dotenv==1.0.0 && \
    python -c "from dotenv import load_dotenv; print('python-dotenv OK')"

# Now install the rest of the packages
RUN pip install --no-cache-dir \
    aiohttp==3.8.4 \
    aiosignal==1.3.1 \
    asgiref==3.6.0 \
    async-timeout==4.0.2 \
    attrs==22.2.0 \
    Babel==2.10.3 \
    bcrypt==3.2.2 \
    certifi==2022.9.24 \
    chardet==5.1.0 \
    charset-normalizer==3.0.1 \
    cryptography==38.0.4 \
    defusedxml==0.7.1 \
    Django==3.2.19 \
    frozenlist==1.3.3 \
    html5lib==1.1 \
    httplib2==0.20.4 \
    idna==3.3 \
    importlib-metadata==4.12.0 \
    Jinja2==3.1.2 \
    markdown-it-py==2.1.0 \
    MarkupSafe==2.1.2 \
    mdurl==0.1.2 \
    more-itertools==8.10.0 \
    multidict==6.0.4 \
    oauthlib==3.2.2 \
    packaging==23.0 \
    pyparsing==3.0.9 \
    PySocks==1.7.1 \
    PyYAML==6.0 \
    requests==2.28.1 \
    requests-toolbelt==0.10.1 \
    rich==13.3.1 \
    six==1.16.0 \
    sqlparse==0.4.2 \
    urllib3==1.26.12 \
    webencodings==0.5.1 \
    yarl==1.8.2 \
    zipp==1.0.0

COPY . .
RUN mkdir -p features

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]