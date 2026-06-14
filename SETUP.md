# Setup

This service runs a JSON REST API, a GPU-backed Celery worker, Redis, and Flower for queue observability. It is designed for an Ubuntu Linux VPS with NVIDIA GPUs and local model files. No paid API, subscription, hosted inference endpoint, or runtime API key is required.

## 1. VPS Prerequisites

- Ubuntu 22.04 or 24.04.
- NVIDIA GPU attached to the VPS. Confirm with `lspci | grep -i nvidia`.
- NVIDIA driver installed and visible through `nvidia-smi`. Do not install only
  `nvidia-utils-*`; install the matching full NVIDIA driver package or use your
  VPS provider's GPU driver image.
- Docker Engine and Docker Compose plugin.
- NVIDIA Container Toolkit configured for Docker.
- Large local storage for model weights. LTX-2.3 assets are large; keep at least several hundred GB free for weights, cache, temp renders, and final videos.
- Large Docker storage. The CUDA base image plus PyTorch CUDA wheels require many GB during build. Keep at least 60-100 GB free on the filesystem that backs Docker, and more if Docker images, build cache, or model files share the same disk.
- A powerful GPU setup. For the default high-quality LTX-2.3 path, prefer datacenter-class GPUs or high-VRAM workstation GPUs. Smaller cards may need `LTX_PIPELINE=distilled`, `LTX_QUANTIZATION=fp8-cast`, and `LTX_OFFLOAD_MODE=cpu`.

## 2. Configure Environment

```bash
cp .env.example .env
```

Review these values in `.env`:

- `HOST_WEB_ROOT=/var/www/html` mounts the host web root into the containers.
- `HOST_MODEL_ROOT=./models` controls where model weights live on the host. In production, put this on a large disk outside the webroot, for example `/mnt/ai-models/text-to-video-generator`.
- `OUTPUT_ALLOWED_ROOTS=/var/www/html,/app/output` controls where API payloads may write video files.
- `INSTALL_LTX=true` clones and installs the official LTX-2 repository during Docker build.
- `LTX_REF=main` selects the LTX-2 git branch, tag, or commit to clone.
- `LTX_UV_SYNC_ARGS=--frozen --no-dev --package ltx-pipelines` installs only the LTX inference pipeline dependencies instead of trainer/development dependencies.
- `LTX_PYTHON=/opt/LTX-2/.venv/bin/python` points the worker at the LTX virtualenv.
- `FLOWER_BASIC_AUTH=user:strong-password` can be added to protect the Flower dashboard. Leave the variable unset to disable Flower auth; an empty value can cause Flower to prompt but reject all credentials on some versions.

## 3. Download Model Assets

Install the Hugging Face CLI if you want the helper script to download the Gemma text encoder:

```bash
sudo apt install -y python3-venv
chmod +x scripts/download-ltx23-models.sh
./scripts/download-ltx23-models.sh
```

If the Hugging Face `hf` CLI is not already installed, the script creates a local helper
virtualenv at `.download-venv` and installs `huggingface_hub` there. It does not
modify the system Python environment.

Gemma requires accepting Google's model terms on Hugging Face before the files
can be downloaded. Open the model page while logged in, accept the terms, then
authenticate the helper CLI:

```bash
.download-venv/bin/hf auth login
./scripts/download-ltx23-models.sh
./scripts/check-model-assets.sh
```

You can also run the downloader non-interactively with a read token:

```bash
HF_TOKEN=hf_xxx ./scripts/download-ltx23-models.sh
```

That credential is only for downloading gated weights; the application does not
use Hugging Face or any hosted inference API at runtime.

The expected model paths are:

```text
models/LTX-2.3/ltx-2.3-22b-dev.safetensors
models/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors
models/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors
models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
models/gemma-3-12b-it-qat-q4_0-unquantized/
```

Verify the host-side paths before starting generation:

```bash
chmod +x scripts/check-model-assets.sh
./scripts/check-model-assets.sh
```

The Compose file mounts `${HOST_MODEL_ROOT}` to container `/models` as read-only.
If your project is under `/var/www/html`, do not keep 100GB+ of model weights in
the webroot on a small root partition. Move the weights to a larger disk and set
`HOST_MODEL_ROOT` in `.env`, for example:

```bash
HOST_MODEL_ROOT=/mnt/ai-models/text-to-video-generator
```

The in-container `LTX_*` paths should usually stay as `/models/...`.

You can also verify what the worker sees inside the container:

```bash
docker compose exec worker ls -lah /models/LTX-2.3
docker compose exec worker ls -lah /models/gemma-3-12b-it-qat-q4_0-unquantized
```

If a model host asks you to accept a model license, do that once in your model hosting account and place the files under `models/`. The application itself still runs fully local and does not call a hosted inference API.

## 4. Build and Start

Direct VPS commands to build and run the application:

```bash
# download the models using the hugging face token
HF_TOKEN=hf_sampleTOKEN HOST_MODEL_ROOT=/mnt/ai-models/text-to-video-generator ./scripts/download-ltx23-models.sh

# verify
HOST_MODEL_ROOT=/mnt/ai-models/text-to-video-generator ./scripts/check-model-assets.sh

# build
docker compose build api

# run
docker compose up -d
```

##### ALTERNATIVELY, USE THE STEPS BELOW:

First verify Docker can see the GPU:

```bash
lspci | grep -i nvidia
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If `nvidia-smi` is missing, install a full NVIDIA driver first. On Ubuntu, start
with the recommended driver discovery command:

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices
```

Install the recommended server/datacenter driver from that output, or use the
driver branch recommended by your VPS provider. The CUDA 12.8.1 image used by
this project needs a recent R570+ compatible driver; older branches such as 535
or 550 are usually not sufficient for the default image.

If the second command fails with `could not select device driver "" with
capabilities: [[gpu]]`, the NVIDIA driver may be installed on the host, but
Docker is not configured with NVIDIA Container Toolkit yet. Install/configure it
from NVIDIA's official guide, then restart Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Build and start the production GPU service:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build api
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

For API/queue smoke tests without GPU, set `GENERATOR_BACKEND=fallback` in
`.env` and run the base Compose file:

```bash
docker compose build api
docker compose up -d
```

If Compose fails with only `failed to execute bake: exit status 1`, rerun with
plain progress and build delegation disabled so Docker prints the real failing
step:

```bash
COMPOSE_BAKE=false docker compose build --no-cache --progress=plain api
```

If the failing step is the LTX install block, the Dockerfile runs:

```bash
uv sync --no-cache --frozen --no-dev --package ltx-pipelines
```

This installs the runtime pipeline package only. To match LTX's full upstream
quick-start environment instead, set this in `.env` before rebuilding:

```bash
LTX_UV_SYNC_ARGS=--frozen
```

If the error contains `No space left on device`, Docker's storage filesystem is
full. Check:

```bash
df -h
docker system df
docker builder du
```

After confirming you do not need old containers/images/build cache, reclaim
space with:

```bash
docker compose down
docker builder prune
docker image prune
```

For an aggressive cleanup of all unused Docker data:

```bash
docker system prune -a
```

Do not run prune commands if this VPS hosts other Docker workloads you need to
keep. A more durable fix is moving Docker's data root to a larger disk, for
example by setting `"data-root": "/mnt/docker-data"` in `/etc/docker/daemon.json`
and restarting Docker.

The API, worker, and Flower services use the same image, so building `api` also
builds the image used by the worker and Flower.

The Docker image installs API, Celery, and Flower dependencies into `/opt/app-venv`.
This avoids Ubuntu/Debian's system-managed `pip`, which cannot always be upgraded
or uninstalled from inside a container because it was installed by `apt`.

Open:

- API docs: `http://YOUR_SERVER:8000/docs`
- Health check: `http://YOUR_SERVER:8000/api/v1/health`
- Flower queue dashboard: `http://YOUR_SERVER:5555`

For production, put Nginx or Caddy in front of the API with TLS and restrict Flower to VPN, localhost, or basic auth.

## 5. Submit a Generation Job

```bash
curl -X POST "http://YOUR_SERVER:8000/api/v1/video-generations" \
  -H "Content-Type: application/json" \
  --data '{
    "prompts": [
      "A chef places fresh chopped vegetables into a hot pan in a clean modern kitchen, steam rising under soft studio lighting.",
      "The same chef gently stirs the vegetables, adds herbs, and plates the dish on a white ceramic plate."
    ],
    "asset_directory": "/var/www/html/sample_project/assets/videos",
    "asset_subdirectory": "",
    "naming_prefix": "recipe-demo"
  }'
```

The response returns immediately with `job_id` and `status_url`. Poll the status URL or inspect the task in Flower.

Output filenames use zero-based prompt indexes by default:

```text
/var/www/html/sample_project/assets/videos/recipe-demo-0.mp4
/var/www/html/sample_project/assets/videos/recipe-demo-1.mp4
```

Set `OUTPUT_FILE_INDEX_BASE=1` if you prefer `recipe-demo-1.mp4`, `recipe-demo-2.mp4`, etc.

## 6. Run Tests

Inside the built image:

```bash
docker compose run --rm api pytest -q
```

A quick local syntax check can be run without dependencies:

```bash
python -m compileall app tests
```

## 7. Scaling Notes

- Keep Celery worker concurrency at `1` per GPU. Video diffusion models load large weights and do not benefit from multiple concurrent jobs on the same GPU unless you have explicitly tested memory headroom.
- Run one worker replica per GPU and pin `NVIDIA_VISIBLE_DEVICES` per replica when scaling.
- Keep Redis persistent volume enabled so queued jobs survive service restarts.
- Use `LTX_OFFLOAD_MODE=cpu` only when necessary; it saves VRAM but slows generation.

## 8. Troubleshooting Corrupt Model Downloads

If Flower shows:

```text
LTX model asset directories are empty. Re-download: /models/gemma-3-12b-it-qat-q4_0-unquantized
```

the Gemma directory exists but contains no downloaded files. Accept the Gemma
terms on Hugging Face, log in with the helper CLI, and download again:

```bash
rm -rf models/gemma-3-12b-it-qat-q4_0-unquantized
./scripts/download-ltx23-models.sh
./scripts/check-model-assets.sh
docker compose restart worker
```

If Flower shows a safetensors error such as:

```text
Error while deserializing header: incomplete metadata, file not fully covered
```

one of the `.safetensors` files exists but is incomplete or corrupt. This can
happen after an interrupted model download. Re-run:

```bash
./scripts/download-ltx23-models.sh
./scripts/check-model-assets.sh
```

The downloader validates existing safetensors files and removes/re-downloads
invalid ones. If you want to force a clean retry manually:

```bash
rm -f models/LTX-2.3/*.partial
rm -f models/LTX-2.3/ltx-2.3-22b-dev.safetensors
./scripts/download-ltx23-models.sh
./scripts/check-model-assets.sh
```

No Docker rebuild is required after replacing model files, because `./models` is
mounted into the worker container.
