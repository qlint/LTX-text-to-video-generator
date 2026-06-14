#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-${HOST_MODEL_ROOT:-${ROOT_DIR}/models}}"
GEMMA_DIR="${MODEL_ROOT}/gemma-3-12b-it-qat-q4_0-unquantized"

required_paths=(
  "${GEMMA_DIR}"
  "${MODEL_ROOT}/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
  "${MODEL_ROOT}/LTX-2.3/ltx-2.3-22b-dev.safetensors"
  "${MODEL_ROOT}/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
  "${MODEL_ROOT}/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
)

validate_safetensors() {
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
    print(exc)
    sys.exit(1)
PY
}

missing=0
for path in "${required_paths[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "MISSING ${path}"
    missing=1
  elif [[ "${path}" == *.safetensors ]]; then
    if error="$(validate_safetensors "${path}")"; then
      echo "OK      ${path}"
    else
      echo "INVALID ${path}: ${error}"
      missing=1
    fi
  elif [ "${path}" = "${GEMMA_DIR}" ]; then
    if [ ! -f "${path}/config.json" ]; then
      echo "INVALID ${path}: missing config.json"
      missing=1
    elif [ ! -f "${path}/model.safetensors.index.json" ]; then
      echo "INVALID ${path}: missing model.safetensors.index.json"
      missing=1
    elif [ "$(find "${path}" -maxdepth 1 -name 'model-*.safetensors' -type f | wc -l)" -lt 1 ]; then
      echo "INVALID ${path}: missing model shard safetensors files"
      missing=1
    else
      echo "OK      ${path}"
    fi
  elif [ -d "${path}" ]; then
    echo "OK      ${path}"
  else
    echo "OK      ${path}"
  fi
done

if [ "${missing}" -ne 0 ]; then
  cat <<MSG

One or more required model assets are missing from ${MODEL_ROOT}.
The Docker Compose file mounts ./models on the host to /models in the containers:

  ./models:/models:ro

Download the assets with:

  ./scripts/download-ltx23-models.sh

or change the Compose volume / LTX_* paths if your model files live elsewhere.
MSG
  exit 1
fi

echo
echo "All required host model paths exist."
