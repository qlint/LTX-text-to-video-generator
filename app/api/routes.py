from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.config import Settings, get_settings
from app.models.schemas import (
    HealthResponse,
    JobStatus,
    VideoGenerationAccepted,
    VideoGenerationRequest,
)
from app.services.job_store import JobStore
from app.services.dependencies import get_job_store
from app.workers.tasks import process_video_generation

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name)


@router.post(
    "/video-generations",
    response_model=VideoGenerationAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_video_generation(
    payload: VideoGenerationRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    job_store: JobStore = Depends(get_job_store),
) -> VideoGenerationAccepted:
    job_id = str(uuid4())
    job_store.create_job(job_id, payload)
    process_video_generation.delay(job_id, payload.model_dump(mode="json"))

    status_path = f"{settings.api_prefix}/video-generations/{job_id}"
    status_url = str(request.base_url).rstrip("/") + status_path
    return VideoGenerationAccepted(
        job_id=job_id,
        status="queued",
        status_url=status_url,
        message="Generation job accepted and queued for background processing.",
    )


@router.get("/video-generations/{job_id}", response_model=JobStatus)
def get_video_generation(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> JobStatus:
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"job_id not found: {job_id}",
        )
    return job

