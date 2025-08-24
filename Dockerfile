FROM python:3.11-slim-bookworm
WORKDIR /app

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        libgtk-3-0 \
        git \
        pkg-config \
        gcc \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.24.3

# Install packages in groups to identify issues
RUN pip install --no-cache-dir \
    aiohttp==3.8.4 \
    aiosignal==1.3.1 \
    asgiref==3.6.0 \
    async-timeout==4.0.2 \
    attrs==22.2.0 \
    Babel==2.10.3 \
    bcrypt==3.2.2 \
    blinker==1.5 \
    certifi==2022.9.24 \
    chardet==5.1.0

RUN pip install --no-cache-dir \
    charset-normalizer==3.0.1 \
    cryptography==38.0.4 \
    defusedxml==0.7.1 \
    Django==3.2.19 \
    frozenlist==1.3.3 \
    html5lib==1.1 \
    httpie==3.2.1 \
    httplib2==0.20.4 \
    idna==3.3

RUN pip install --no-cache-dir \
    importlib-metadata==4.12.0 \
    Jinja2==3.1.2 \
    markdown-it-py==2.1.0 \
    MarkupSafe==2.1.2 \
    mdurl==0.1.2 \
    mechanize==0.4.8 \
    more-itertools==8.10.0 \
    multidict==6.0.4

RUN pip install --no-cache-dir \
    oauthlib==3.2.2 \
    packaging==23.0 \
    Pygments==2.14.0 \
    PyJWT==2.6.0 \
    PyNaCl==1.5.0 \
    pyOpenSSL==23.0.0 \
    pyparsing==3.0.9 \
    PySocks==1.7.1

RUN pip install --no-cache-dir \
    python-dateutil==2.8.2 \
    python-gnupg==0.4.9 \
    python-magic==0.4.26 \
    pytz==2022.7.1 \
    PyYAML==6.0 \
    requests==2.28.1 \
    requests-toolbelt==0.10.1 \
    rich==13.3.1

RUN pip install --no-cache-dir \
    sentry-sdk==1.9.10 \
    six==1.16.0 \
    sqlparse==0.4.2 \
    urllib3==1.26.12 \
    wadllib==1.3.6 \
    webencodings==0.5.1 \
    yarl==1.8.2 \
    zipp==1.0.0

# App files
COPY . .
RUN mkdir -p features

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]