#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-${HOST_MODEL_ROOT:-${ROOT_DIR}/models}}"
MODEL_DIR="${MODEL_DIR:-${MODEL_ROOT}/LTX-2.3}"
GEMMA_DIR="${GEMMA_DIR:-${MODEL_ROOT}/gemma-3-12b-it-qat-q4_0-unquantized}"
GEMMA_REPO="${GEMMA_REPO:-google/gemma-3-12b-it-qat-q4_0-unquantized}"
DOWNLOAD_VENV="${DOWNLOAD_VENV:-${ROOT_DIR}/.download-venv}"

mkdir -p "${MODEL_DIR}" "${GEMMA_DIR}"

if [ -x "${DOWNLOAD_VENV}/bin/hf" ]; then
  PATH="${DOWNLOAD_VENV}/bin:${PATH}"
fi

is_valid_safetensors() {
  local path="$1"
  python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    size = path.stat().st_size
    if size <= 8:
        raise ValueError("file is too small")
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        header_end = 8 + header_size
        if header_size <= 0 or header_end > size:
            raise ValueError("invalid safetensors header length")
        header = json.loads(handle.read(header_size).decode("utf-8"))
    data_size = size - header_end
    tensor_offsets = [
        value.get("data_offsets")
        for key, value in header.items()
        if key != "__metadata__" and isinstance(value, dict)
    ]
    tensor_offsets = [offset for offset in tensor_offsets if offset is not None]
    if not tensor_offsets:
        raise ValueError("no tensor data offsets found")
    max_end = 0
    for offset in tensor_offsets:
        if (
            not isinstance(offset, list)
            or len(offset) != 2
            or offset[0] < 0
            or offset[1] < offset[0]
            or offset[1] > data_size
        ):
            raise ValueError("invalid tensor data offsets")
        max_end = max(max_end, offset[1])
    if max_end != data_size:
        raise ValueError("safetensors data is incomplete or has trailing bytes")
except Exception as exc:
    print(f"INVALID {path}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

download_file() {
  local url="$1"
  local output="$2"
  local partial="${output}.partial"
  if [ -f "${output}" ]; then
    if [[ "${output}" == *.safetensors ]]; then
      if is_valid_safetensors "${output}"; then
        echo "Already exists and passed safetensors validation: ${output}"
        return
      fi
      echo "Removing invalid safetensors file: ${output}"
      rm -f "${output}"
    else
      echo "Already exists: ${output}"
      return
    fi
  fi

  if [ -f "${partial}" ] && [[ "${partial}" == *.safetensors.partial ]]; then
    if ! is_valid_safetensors "${partial}"; then
      echo "Removing invalid partial download: ${partial}"
      rm -f "${partial}"
    fi
  fi

  if [ -f "${partial}" ]; then
    echo "Resuming ${url}"
  else
    echo "Downloading ${url}"
  fi
  curl \
    -L \
    --fail \
    --retry 5 \
    --retry-delay 10 \
    --retry-all-errors \
    --continue-at - \
    --output "${partial}" \
    "${url}"

  if [[ "${output}" == *.safetensors ]]; then
    is_valid_safetensors "${partial}"
  fi
  mv "${partial}" "${output}"
}

gemma_has_required_files() {
  [ -f "${GEMMA_DIR}/config.json" ] \
    && [ -f "${GEMMA_DIR}/model.safetensors.index.json" ] \
    && [ "$(find "${GEMMA_DIR}" -maxdepth 1 -name 'model-*.safetensors' -type f | wc -l)" -ge 1 ]
}

hf_cli_path() {
  command -v hf 2>/dev/null || printf "%s/bin/hf" "${DOWNLOAD_VENV}"
}

print_gemma_help() {
  cat <<MSG

Gemma text encoder files were not downloaded into:

  ${GEMMA_DIR}

The ${GEMMA_REPO} repository is gated on Hugging Face. Open this page in a
browser while logged in, review, and accept Google's Gemma usage license:

  https://huggingface.co/${GEMMA_REPO}

Then authenticate the local helper CLI and retry:

  $(hf_cli_path) auth login
  ./scripts/download-ltx23-models.sh
  ./scripts/check-model-assets.sh

Alternatively run non-interactively with a read token:

  HF_TOKEN=hf_xxx ./scripts/download-ltx23-models.sh

That token is only used for downloading gated weights. The application does not
call Hugging Face or any hosted inference API at runtime.
MSG
}

download_file \
  "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors" \
  "${MODEL_DIR}/ltx-2.3-22b-dev.safetensors"

download_file \
  "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-1.1.safetensors" \
  "${MODEL_DIR}/ltx-2.3-22b-distilled-1.1.safetensors"

download_file \
  "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384-1.1.safetensors" \
  "${MODEL_DIR}/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

download_file \
  "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
  "${MODEL_DIR}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found. Creating a local helper venv at ${DOWNLOAD_VENV}"
  python3 -m venv "${DOWNLOAD_VENV}"
  "${DOWNLOAD_VENV}/bin/python" -m pip install --upgrade pip
  "${DOWNLOAD_VENV}/bin/python" -m pip install --upgrade huggingface_hub
  PATH="${DOWNLOAD_VENV}/bin:${PATH}"
fi

if command -v hf >/dev/null 2>&1; then
  echo "Downloading Gemma text encoder into ${GEMMA_DIR}"
  if ! hf download "${GEMMA_REPO}" --local-dir "${GEMMA_DIR}"; then
    print_gemma_help
    exit 1
  fi
  if ! gemma_has_required_files; then
    print_gemma_help
    exit 1
  fi
else
  cat <<MSG
Could not install or find the Hugging Face hf CLI. Install it manually to
download the Gemma text encoder:

  python3 -m pip install -U huggingface_hub
  hf download ${GEMMA_REPO} --local-dir "${GEMMA_DIR}"

No paid API, subscription, or runtime key is used by this service.
MSG
  exit 1
fi
