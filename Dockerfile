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

RUN pip install --upgrade pip

# Install packages in logical groups to avoid conflicts

# 1. Foundational numerical packages
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    Pillow==10.0.1

# 2. Machine learning core
RUN pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    joblib==1.3.2

# 3. Computer vision
RUN pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78

# 4. Deep learning (use CPU version for stability)
RUN pip install --no-cache-dir \
    tensorflow-cpu==2.13.0

# 5. Database
RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.7

# 6. Web framework
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.22.0

# 7. Utilities
RUN pip install --no-cache-dir \
    python-dotenv==1.0.0

# 8. HTTP/Networking
RUN pip install --no-cache-dir \
    aiohttp==3.8.4 \
    aiosignal==1.3.1 \
    requests==2.28.1 \
    httplib2==0.20.4

# 9. Security/Crypto
RUN pip install --no-cache-dir \
    cryptography==38.0.4 \
    bcrypt==3.2.2 \
    PyNaCl==1.5.0 \
    pyOpenSSL==23.0.0

# 10. Web utilities
RUN pip install --no-cache-dir \
    html5lib==1.1 \
    defusedxml==0.7.1 \
    Babel==2.10.3 \
    Jinja2==3.1.2

# 11. Data processing
RUN pip install --no-cache-dir \
    pandas==2.0.3 \
    python-dateutil==2.8.2 \
    pytz==2022.7.1

# 12. Async utilities
RUN pip install --no-cache-dir \
    async-timeout==4.0.2 \
    asgiref==3.6.0

# 13. Text processing
RUN pip install --no-cache-dir \
    chardet==5.1.0 \
    charset-normalizer==3.0.1 \
    markdown-it-py==2.1.0

# 14. Other utilities
RUN pip install --no-cache-dir \
    attrs==22.2.0 \
    certifi==2022.9.24 \
    frozenlist==1.3.3 \
    idna==3.3 \
    importlib-metadata==4.12.0 \
    more-itertools==8.10.0 \
    multidict==6.0.4 \
    oauthlib==3.2.2 \
    packaging==23.0 \
    pyparsing==3.0.9 \
    PySocks==1.7.1 \
    PyYAML==6.0 \
    requests-toolbelt==0.10.1 \
    rich==13.3.1 \
    six==1.16.0 \
    urllib3==1.26.12 \
    webencodings==0.5.1 \
    yarl==1.8.2 \
    zipp==1.0.0

# 15. Django and related (install together)
RUN pip install --no-cache-dir \
    Django==3.2.19 \
    sqlparse==0.4.2

# 16. Remaining packages
RUN pip install --no-cache-dir \
    Pygments==2.14.0 \
    PyJWT==2.6.0 \
    MarkupSafe==2.1.2 \
    mdurl==0.1.2 \
    wadllib==1.3.6 \
    sentry-sdk==1.9.10

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
from dotenv import load_dotenv; print('python-dotenv OK'); \
print('All critical imports successful!')\
"

COPY . .
RUN mkdir -p features

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]