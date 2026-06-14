from __future__ import annotations

from functools import lru_cache

from redis import Redis

from app.core.config import Settings, get_settings
from app.services.job_store import JobStore


@lru_cache
def get_redis_client() -> Redis:
    settings = get_settings()
    return Redis.from_url(settings.redis_url)


def get_job_store() -> JobStore:
    settings: Settings = get_settings()
    return JobStore(get_redis_client(), settings)
