from __future__ import annotations

import logging
from typing import Any

from redis import Redis

from app.core.config import get_settings
from app.inference.factory import build_video_generator
from app.models.schemas import VideoGenerationRequest
from app.queue.celery_app import celery_app
from app.services.job_store import JobStore
from app.services.output_paths import OutputPathResolver
from app.services.prompt_planner import PromptPlanner
from app.services.video_orchestrator import VideoOrchestrator
from app.utils.ffmpeg import FFmpeg

logger = logging.getLogger(__name__)


@celery_app.task(name="video_generation.process", bind=True)
def process_video_generation(self: Any, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    redis_client = Redis.from_url(settings.redis_url)
    job_store = JobStore(redis_client, settings)
    request = VideoGenerationRequest.model_validate(payload)

    ffmpeg = FFmpeg(settings)
    generator = build_video_generator(settings, ffmpeg)
    orchestrator = VideoOrchestrator(
        settings=settings,
        generator=generator,
        ffmpeg=ffmpeg,
        resolver=OutputPathResolver(settings),
        planner=PromptPlanner(settings),
    )

    try:
        job_store.mark_running(job_id)
        outputs = orchestrator.generate_batch(job_id, request, job_store)
        job_store.mark_completed(job_id)
        return {"job_id": job_id, "outputs": [str(path) for path in outputs]}
    except Exception as exc:
        logger.exception("video generation job failed", extra={"job_id": job_id})
        job_store.mark_failed(job_id, str(exc))
        raise
