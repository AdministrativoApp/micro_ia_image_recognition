FROM python:3.10-slim-bookworm
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=180

# Fail fast if not amd64 (needed for MP wheels)
RUN uname -m | grep -qE 'x86_64|amd64' || (echo "ERROR: Build must target linux/amd64"; exit 1)

# System deps for TF/MP/OpenCV + headers for cffi/cryptography
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    git pkg-config gcc g++ python3-dev \
    libffi-dev libssl-dev \
 && rm -rf /var/lib/apt/lists/*

# (Optional) print arch/Python
RUN python - <<'PY'
import platform, sys
print("ARCH:", platform.machine(), "PY:", sys.version)
PY

# Tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Core compatible pins
RUN pip install "numpy==1.24.3" "protobuf>=3.20.3,<5"

# TensorFlow first
RUN pip install "tensorflow-cpu==2.15.0.post1" && \
    python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"

RUN pip install python-multipart && \
    python -c "import multipart; print('python-multipart OK')"

# MediaPipe (cp310 wheel)
# Let MP pull its own compatible OpenCV
RUN pip install --only-binary=:all: "mediapipe==0.10.21" -vv && \
    python -c "import mediapipe as mp; print('MediaPipe', mp.__version__)"

# Main Python stack
RUN pip install \
    scipy==1.11.3 Pillow==10.0.1 scikit-learn==1.3.2 joblib==1.3.2 && \
    python - <<'PY'
import scipy, PIL, sklearn, joblib
print("SciPy", scipy.__version__)
print("Pillow", PIL.__version__)
print("Scikit-learn", sklearn.__version__)
print("Joblib", joblib.__version__)
PY

RUN pip install \
    psycopg2-binary==2.9.7 fastapi==0.104.1 uvicorn==0.22.0 python-dotenv==1.0.0 && \
    python - <<'PY'
import psycopg2, fastapi, uvicorn
from dotenv import load_dotenv
print("psycopg2", psycopg2.__version__)
print("fastapi", fastapi.__version__)
print("uvicorn", uvicorn.__version__)
print("python-dotenv OK")
PY

# --- Hardened extras install ---
# 1) Try each package with wheels-only; collect failures.
# 2) If any failed, install Rust and retry ONLY those (verbose).
RUN set -eux; \
  PKGS="\
    aiohttp==3.8.4 aiosignal==1.3.1 asgiref==3.6.0 async-timeout==4.0.2 attrs==22.2.0 \
    Babel==2.10.3 bcrypt==4.1.2 certifi==2022.9.24 chardet==5.1.0 \
    charset-normalizer==3.0.1 cryptography==41.0.7 defusedxml==0.7.1 \
    Django==3.2.19 frozenlist==1.3.3 html5lib==1.1 httplib2==0.20.4 \
    idna==3.3 Jinja2==3.1.2 markdown-it-py==2.1.0 \
    MarkupSafe==2.1.2 mdurl==0.1.2 more-itertools==8.10.0 multidict==6.0.4 \
    oauthlib==3.2.2 packaging==23.0 pyparsing==3.0.9 PySocks==1.7.1 \
    PyYAML==6.0 requests==2.28.1 requests-toolbelt==0.10.1 rich==13.3.1 \
    six==1.16.0 sqlparse==0.4.2 urllib3==1.26.12 webencodings==0.5.1 yarl==1.8.2 zipp==1.0.0 \
  "; \
  FAILS=""; \
  for P in $PKGS; do \
    echo ">>> Trying wheel for $P"; \
    if ! pip install --only-binary=:all: --prefer-binary "$P"; then \
      echo "!!! No wheel for $P"; \
      FAILS="$FAILS $P"; \
    fi; \
  done; \
  if [ -n "$FAILS" ]; then \
    echo ">>> Installing Rust toolchain for source builds…"; \
    apt-get update -y && apt-get install -y --no-install-recommends rustc cargo && rm -rf /var/lib/apt/lists/*; \
    for P in $FAILS; do \
      echo ">>> Retrying from source (verbose): $P"; \
      pip install -vv --prefer-binary "$P"; \
    done; \
  fi

# App
COPY . .
RUN mkdir -p features

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
