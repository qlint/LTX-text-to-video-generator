from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from redis import Redis

from app.core.config import Settings
from app.models.schemas import JobStatus, VideoGenerationRequest, VideoOutput


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    """Small Redis-backed status store for queued generation jobs."""

    def __init__(self, redis_client: Redis, settings: Settings) -> None:
        self.redis = redis_client
        self.settings = settings

    def create_job(self, job_id: str, request: VideoGenerationRequest) -> None:
        now = utc_now_iso()
        outputs = [
            VideoOutput(prompt_index=index).model_dump(mode="json")
            for index, _ in enumerate(request.prompts)
        ]
        data = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "total_prompts": len(request.prompts),
            "completed_prompts": 0,
            "outputs": json.dumps(outputs),
            "request": request.model_dump_json(),
            "error": "",
        }
        key = self._key(job_id)
        self.redis.hset(key, mapping=data)
        self.redis.expire(key, self.settings.job_ttl_seconds)

    def mark_running(self, job_id: str) -> None:
        self._update(job_id, {"status": "running"})

    def mark_prompt_running(self, job_id: str, prompt_index: int, path: str) -> None:
        outputs = self._outputs(job_id)
        outputs[prompt_index] = VideoOutput(
            prompt_index=prompt_index,
            path=path,
            status="running",
        ).model_dump(mode="json")
        self._update(job_id, {"outputs": json.dumps(outputs)})

    def mark_prompt_completed(self, job_id: str, prompt_index: int, path: str) -> None:
        outputs = self._outputs(job_id)
        outputs[prompt_index] = VideoOutput(
            prompt_index=prompt_index,
            path=path,
            status="completed",
        ).model_dump(mode="json")
        completed = sum(1 for output in outputs if output.get("status") == "completed")
        self._update(
            job_id,
            {
                "outputs": json.dumps(outputs),
                "completed_prompts": completed,
            },
        )

    def mark_prompt_failed(self, job_id: str, prompt_index: int, path: str, error: str) -> None:
        outputs = self._outputs(job_id)
        outputs[prompt_index] = VideoOutput(
            prompt_index=prompt_index,
            path=path,
            status="failed",
            error=error,
        ).model_dump(mode="json")
        self._update(job_id, {"outputs": json.dumps(outputs)})

    def mark_completed(self, job_id: str) -> None:
        self._update(job_id, {"status": "completed"})

    def mark_failed(self, job_id: str, error: str) -> None:
        self._update(job_id, {"status": "failed", "error": error})

    def get_job(self, job_id: str) -> JobStatus | None:
        raw = self.redis.hgetall(self._key(job_id))
        if not raw:
            return None

        data = {self._decode(k): self._decode(v) for k, v in raw.items()}
        outputs = json.loads(data.get("outputs") or "[]")
        request = json.loads(data["request"]) if data.get("request") else None
        error = data.get("error") or None

        return JobStatus(
            job_id=data["job_id"],
            status=data["status"],
            created_at=self._parse_datetime(data.get("created_at")),
            updated_at=self._parse_datetime(data.get("updated_at")),
            total_prompts=int(data.get("total_prompts") or 0),
            completed_prompts=int(data.get("completed_prompts") or 0),
            outputs=outputs,
            error=error,
            request=request,
        )

    def _outputs(self, job_id: str) -> list[dict[str, Any]]:
        raw = self.redis.hget(self._key(job_id), "outputs")
        if raw is None:
            raise KeyError(f"unknown job_id: {job_id}")
        return json.loads(self._decode(raw))

    def _update(self, job_id: str, mapping: dict[str, Any]) -> None:
        mapping["updated_at"] = utc_now_iso()
        key = self._key(job_id)
        self.redis.hset(key, mapping=mapping)
        self.redis.expire(key, self.settings.job_ttl_seconds)

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value)

    @staticmethod
    def _decode(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _key(job_id: str) -> str:
        return f"ttv:job:{job_id}"

