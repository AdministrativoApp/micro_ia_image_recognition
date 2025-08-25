# -------- Base --------
FROM python:3.10-slim-bookworm
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Fail fast if not amd64 (MediaPipe 0.10.0 wheel requires x86_64)
RUN uname -m | grep -qE 'x86_64|amd64' || (echo "ERROR: Build must target linux/amd64. In GitHub Actions, set platforms: linux/amd64"; exit 1)

# -------- System deps needed by OpenCV/MediaPipe/TensorFlow --------
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        git \
        pkg-config \
        gcc \
        g++ \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# (Optional) Show arch & Python early in build logs
RUN python - <<'PY'
import platform, sys
print("ARCH:", platform.machine(), "PY:", sys.version)
PY

# -------- Python packaging tools --------
RUN python -m pip install --upgrade pip setuptools wheel

# -------- Core pins compatible across TF 2.13 + MP 0.10.0 --------
# numpy first helps manylinux wheel resolution; protobuf pinned for TF/MP; absl<2 required by TF 2.13
RUN pip install \
    "numpy==1.24.3" \
    "protobuf>=3.20.3,<5" \
    "absl-py==1.4.0"

# -------- TensorFlow FIRST (locks absl<2) --------
RUN pip install "tensorflow-cpu==2.13.0" && \
    python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"

# -------- MediaPipe (wheel only) --------
# DO NOT install a separate opencv-python; let mediapipe pull a compatible one
RUN pip install --only-binary=:all: "mediapipe==0.10.0" && \
    python -c "import mediapipe as mp; print('MediaPipe', mp.__version__)"

# -------- Your main Python stack --------
RUN pip install \
    scipy==1.11.3 \
    Pillow==10.0.1 \
    scikit-learn==1.3.2 \
    joblib==1.3.2 && \
    python - <<'PY'
import scipy, PIL, sklearn, joblib
print("SciPy", scipy.__version__)
print("Pillow", PIL.__version__)
print("Scikit-learn", sklearn.__version__)
print("Joblib", joblib.__version__)
PY

RUN pip install \
    psycopg2-binary==2.9.7 \
    fastapi==0.104.1 \
    uvicorn==0.22.0 \
    python-dotenv==1.0.0 && \
    python - <<'PY'
import psycopg2, fastapi, uvicorn
from dotenv import load_dotenv
print("psycopg2", psycopg2.__version__)
print("fastapi", fastapi.__version__)
print("uvicorn", uvicorn.__version__)
print("python-dotenv OK")
PY

# -------- Extra libs you listed (keep only what you truly need) --------
RUN pip install \
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
    frozenlist==1.3.3

RUN pip install \
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
    packaging==23.0

RUN pip install \
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

# -------- App files --------
COPY . .
RUN mkdir -p features

# -------- Runtime --------
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
