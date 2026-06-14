#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000/api/v1}"
OUTPUT_DIR="${OUTPUT_DIR:-/var/www/html/sample_project/assets/videos}"

curl -sS -X POST "${API_BASE}/video-generations" \
  -H "Content-Type: application/json" \
  --data @- <<JSON
{
  "prompts": [
    "A chef places fresh chopped vegetables into a hot pan in a clean modern kitchen, steam rising under soft studio lighting.",
    "The same chef gently stirs the vegetables, adds herbs, and plates the dish on a white ceramic plate."
  ],
  "asset_directory": "${OUTPUT_DIR}",
  "asset_subdirectory": "",
  "naming_prefix": "recipe-demo"
}
JSON

