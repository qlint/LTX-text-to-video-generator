# Technical Considerations

## Decision Summary

The default backend is LTX-2.3 through the official Lightricks LTX-2 Python pipeline. The service wraps it with FastAPI, Celery, Redis, Flower, Docker, ffmpeg post-processing, guarded filesystem writes, and prompt/segment planning for long-form output.

Primary references:

- LTX open-source documentation: https://docs.ltx.video/open-source-model/getting-started/overview
- LTX-2.3 model page: https://ltx.io/model/ltx-2-3
- LTX-2 GitHub repository: https://github.com/Lightricks/LTX-2
- LTX-2.3 Hugging Face model card: https://huggingface.co/Lightricks/LTX-2.3
- LTX-2 technical report: https://arxiv.org/abs/2601.03233
- Wan2.2 repository: https://github.com/Wan-Video/Wan2.2

## Why LTX-2.3

LTX-2.3 is suitable for this build because it directly matches the hard requirements better than the current open alternatives:

- It is an audio-video foundation model, so sound is generated in the same model path instead of being bolted on after silent video generation.
- It has open weights and official local Python inference code.
- The official LTX-2 repository includes production-oriented two-stage pipelines, HQ mode, distilled mode, fp8 quantization options, offload modes, prompt enhancement, spatial upscaling, and LoRA hooks.
- It supports 1080p-class output. The service generates at model-friendly dimensions such as `1920x1088`, then crops/normalizes to `1920x1080`.
- It supports portrait-oriented workflows, so short-form vertical output can be inferred from prompts containing words like `vertical`, `portrait`, `reel`, `shorts`, or `TikTok`.

The important caveat is licensing. LTX-2.3 has no usage-metered runtime cost in this self-hosted design, but operators must review the LTX community license for their business use case.
The Gemma text encoder is also license-gated on Hugging Face; accepting that
license and using a read token may be required for one-time weight download, but
no hosted inference API is used at runtime.

## Why Not Wan2.2 as the Primary Backend

Wan2.2 is excellent and open. It is a strong candidate for a future secondary adapter, especially when visual-only 720p generation is acceptable. It was not chosen as the primary backend because the repository documents 720p/24fps text-to-video paths and separate speech/audio-driven paths, while this project asks for 1080p minimum, 30fps output, long/short form, and sound when needed. Wan2.2 also recommends optional prompt extension using either Dashscope API or a local Qwen model; the API option conflicts with the no-API-key requirement.

## Long-Form Strategy

Open video diffusion models still have native clip windows. LTX docs describe generation around short clips rather than arbitrary feature-length video in one pass. This application handles long-form generation by:

1. Estimating duration from prompt length.
2. Splitting long prompts into segment prompts.
3. Injecting a shared batch context into every segment so scene, lighting, subject identity, wardrobe, camera style, and sound remain consistent.
4. Rendering each segment with LTX.
5. Normalizing segments to the same codec, resolution, and FPS.
6. Concatenating segments into one MP4 per original prompt.

`MAX_AUTO_SEGMENTS=0` means the application does not impose an artificial segment-count cap. Operators can set a cap to control cost and queue time.

## Resolution and FPS

The service targets final MP4 files at:

- Landscape: `1920x1080`
- Portrait: `1080x1920`
- FPS: `30`
- Video codec: H.264, yuv420p, CRF 18
- Audio codec: AAC, 192 kbps

LTX inference uses dimensions divisible by the model requirements, for example `1920x1088` for landscape. ffmpeg then crops to exact 1080p. The worker can generate at `48fps` and normalize to `30fps`, which gives a clean final file while staying close to LTX's preferred frame-rate modes.

## Queue and Observability

Celery + Redis + Flower was selected because:

- It is free and self-hosted.
- Redis is simple to operate on a single VPS.
- Celery supports immediate API acknowledgement with background task execution.
- Flower provides a browser dashboard for task state, runtime, failures, retries, and worker health.
- Worker concurrency can be kept at `1` to protect GPU memory.

The API stores job status separately in Redis hashes so clients do not need to understand Celery internals.

## Filesystem Safety

The required payload allows clients to specify an output directory. That is useful for integration with existing PHP/Apache/Nginx sites, but dangerous if exposed directly. The service therefore:

- Requires `asset_directory` to be absolute.
- Allows `asset_subdirectory` only as a relative path.
- Resolves paths before writing.
- Rejects writes outside `OUTPUT_ALLOWED_ROOTS` unless explicitly disabled.
- Sanitizes `naming_prefix` for filesystem-safe filenames.
- Uses temp files and atomic replacement so clients never see half-written final files.

## Data Lifecycle

- The API validates the JSON payload and stores a queued job in Redis.
- The worker creates output directories only after path validation.
- Raw model output is written into a hidden temporary directory inside the target output directory.
- ffmpeg-normalized segments are also temporary.
- The final MP4 is moved into place atomically.
- Temporary files are removed automatically after each prompt.
- Job metadata expires after `JOB_TTL_SECONDS`.

## VPS Optimization

Recommended settings for a powerful NVIDIA VPS:

- Use NVIDIA Container Toolkit and verify `docker run --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`.
- Start GPU-backed generation with `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`.
- Keep `WORKER_CONCURRENCY=1` or the provided Celery `--concurrency=1`.
- Use `ipc: host` and `shm_size: 32gb` for large CUDA workloads.
- Start with `LTX_PIPELINE=hq`, `LTX_QUANTIZATION=fp8-cast`, `LTX_OFFLOAD_MODE=none`.
- The Docker build uses `uv sync --no-cache --frozen --no-dev --package ltx-pipelines` so the image installs inference dependencies without trainer/development packages and without retaining uv's package cache.
- Use `LTX_OFFLOAD_MODE=cpu` only for VRAM pressure.
- Keep model files on fast NVMe storage.
- Monitor `nvidia-smi dmon`, container logs, Redis memory, and disk usage.
- Put the API behind TLS and restrict Flower.

## Limitations

- Prompt-to-video quality is inherently probabilistic. The same prompt can still produce identity drift or motion artifacts.
- Exact continuity across separate prompt items is improved through shared prompt context and seeds, but true identity locking would require image/video conditioning or LoRA assets not present in the required payload.
- Long-form output is stitched from multiple generated clips.
- If the prompt implies precise dialogue, review generated audio manually; open audio-video models can still mispronounce or hallucinate speech.
