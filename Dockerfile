FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    PATH="/opt/app-venv/bin:/root/.local/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m venv /opt/app-venv \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

ARG INSTALL_LTX=false
ARG LTX_REPO_URL=https://github.com/Lightricks/LTX-2.git
ARG LTX_REF=main
ARG LTX_UV_SYNC_ARGS="--frozen --no-dev --package ltx-pipelines"
RUN if [ "${INSTALL_LTX}" = "true" ]; then \
        set -eux; \
        curl -LsSf https://astral.sh/uv/install.sh | sh; \
        uv --version; \
        git clone --depth 1 --branch "${LTX_REF}" "${LTX_REPO_URL}" /opt/LTX-2; \
        cd /opt/LTX-2; \
        env -u VIRTUAL_ENV uv sync --no-cache ${LTX_UV_SYNC_ARGS}; \
        /opt/LTX-2/.venv/bin/python -c "import ltx_core, ltx_pipelines; print('LTX runtime import check passed')"; \
        rm -rf /root/.cache/uv /tmp/uv-cache; \
    fi

COPY app ./app

RUN mkdir -p /app/output /models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
