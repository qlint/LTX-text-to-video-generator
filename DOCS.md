# Software Documentation

## Directory Structure

```text
app/
  api/routes.py                 FastAPI routes.
  core/config.py                Environment-driven settings.
  core/logging.py               JSON logging setup.
  inference/base.py             Generator interface and generation spec.
  inference/factory.py          Backend selection.
  inference/fallback.py         ffmpeg test renderer for development.
  inference/ltx23.py            LTX-2.3 subprocess adapter.
  models/schemas.py            Pydantic request/response models.
  queue/celery_app.py           Celery configuration.
  services/dependencies.py      Redis and JobStore dependency providers.
  services/job_store.py         Redis job status persistence.
  services/output_paths.py      Output path validation and filename rules.
  services/prompt_planner.py    Shared context, orientation, segmentation.
  services/video_orchestrator.py Model execution and artifact lifecycle.
  utils/ffmpeg.py               ffmpeg command wrapper.
  workers/tasks.py              Celery task entrypoint.
scripts/
  download-ltx23-models.sh      Model download helper.
  check-model-assets.sh         Verifies required host-side model paths.
  smoke-test-api.sh             Example API request.
tests/                          Focused unit tests.
Dockerfile                      CUDA-capable application image.
docker-compose.yml              API, worker, Redis, Flower.
```

Model weights are mounted from `${HOST_MODEL_ROOT:-./models}` on the host to
`/models` in the containers. Keep `HOST_MODEL_ROOT` on a large disk in
production; the application code itself is small, but self-hosted video model
weights are large.

## API

### `GET /api/v1/health`

Returns service health.

```json
{
  "status": "ok",
  "service": "Self Hosted Text-to-Video Generator"
}
```

### `POST /api/v1/video-generations`

Accepts the required JSON payload and returns immediately with a queued job.

```json
{
  "prompts": ["...", "..."],
  "asset_directory": "/var/www/html/sample_project/assets/videos",
  "asset_subdirectory": "",
  "naming_prefix": "demo"
}
```

Rules:

- `prompts` must be a non-empty array of non-empty strings.
- Each prompt creates one final MP4.
- `asset_directory` must be absolute.
- `asset_subdirectory` may be `null`, string, number, or empty string. Empty and `null` mean no subdirectory.
- `naming_prefix` may be `null`, string, or number. Empty and `null` mean no prefix.
- Booleans are rejected for `asset_subdirectory` and `naming_prefix`.

Response:

```json
{
  "job_id": "uuid",
  "status": "queued",
  "status_url": "http://server/api/v1/video-generations/uuid",
  "message": "Generation job accepted and queued for background processing."
}
```

### `GET /api/v1/video-generations/{job_id}`

Returns job status, per-prompt output paths, and errors if any.

Statuses:

- `queued`
- `running`
- `completed`
- `failed`

## File Naming

Default index base is zero.

Without prefix:

```text
0.mp4
1.mp4
```

With prefix:

```text
demo-0.mp4
demo-1.mp4
```

`naming_prefix` is sanitized to letters, numbers, `.`, `_`, and `-`. If `OVERWRITE_OUTPUTS=true`, existing files with the same final name are atomically replaced.

## Process Flow

1. `routes.create_video_generation()` validates the payload with `VideoGenerationRequest`.
2. A UUID is created and `JobStore.create_job()` writes queued state to Redis.
3. `process_video_generation.delay()` enqueues the Celery task.
4. `workers.tasks.process_video_generation()` rebuilds dependencies inside the worker process.
5. `VideoOrchestrator.generate_batch()` resolves the output directory and creates prompt plans.
6. `PromptPlanner.plan_batch()` adds a shared scene context and splits long prompts into segments.
7. `Ltx23VideoGenerator.generate()` runs the official LTX pipeline module.
8. `FFmpeg.normalize()` converts each segment to the final resolution/FPS/codec profile.
9. Multiple segments are joined with `FFmpeg.concat()`.
10. The final MP4 is atomically moved into the requested output directory.
11. `JobStore` updates per-prompt and job-level status.

## Main Classes and Functions

### `Settings` in `app/core/config.py`

Central configuration object. Important constants include:

- `generator_backend`: `ltx23` or `fallback`.
- `output_allowed_roots`: comma-separated safe write roots.
- `target_fps`: final output FPS.
- `ltx_pipeline`: `hq`, `standard`, or `distilled`.
- `ltx_model_frame_rate`: frame rate passed to LTX before final normalization.
- `max_ltx_segment_seconds`: maximum native segment duration before splitting.

### `VideoGenerationRequest` in `app/models/schemas.py`

Defines the POST payload. It trims prompts, rejects empty prompt strings, forbids extra fields, and rejects boolean scalar path/prefix values.

### `JobStore` in `app/services/job_store.py`

Stores job state in Redis hashes under `ttv:job:{job_id}`. Methods:

- `create_job()`
- `mark_running()`
- `mark_prompt_running()`
- `mark_prompt_completed()`
- `mark_prompt_failed()`
- `mark_completed()`
- `mark_failed()`
- `get_job()`

### `OutputPathResolver` in `app/services/output_paths.py`

Resolves safe directories and filenames. It enforces absolute `asset_directory`, relative `asset_subdirectory`, allowed roots, directory creation, and sanitized prefixes.

### `PromptPlanner` in `app/services/prompt_planner.py`

Creates `PromptPlan` and `SegmentPlan` objects. It:

- Infers landscape or portrait orientation.
- Injects shared batch context for scene consistency.
- Estimates duration from prompt length.
- Splits long prompts into model-sized chunks.
- Computes LTX-compatible frame counts where `num_frames % 8 == 1`.

### `Ltx23VideoGenerator` in `app/inference/ltx23.py`

Builds a subprocess command for the official LTX pipeline. Supported modules:

- `ltx_pipelines.ti2vid_two_stages_hq`
- `ltx_pipelines.ti2vid_two_stages`
- `ltx_pipelines.distilled`

It validates required model paths and checks safetensors headers before running,
so incomplete model downloads fail with a direct asset error instead of a deep
pipeline traceback.

### `VideoOrchestrator` in `app/services/video_orchestrator.py`

Owns the artifact lifecycle for a batch. It creates temp directories, calls the generator, normalizes segments, concatenates long output, atomically places final files, updates job state, and cleans temporary files.

### `FFmpeg` in `app/utils/ffmpeg.py`

Runs ffmpeg safely with argument lists. It provides:

- `normalize()`
- `concat()`
- `generate_fallback_video()`
- `run()`

## Development Backend

Set `GENERATOR_BACKEND=fallback` to generate a tiny ffmpeg test pattern instead of loading LTX. This is only for API, queue, and filesystem testing. It is not the production text-to-video model.
