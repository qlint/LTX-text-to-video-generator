from __future__ import annotations

from app.core.config import Settings
from app.inference.base import VideoGenerator
from app.inference.fallback import FallbackVideoGenerator
from app.inference.ltx23 import Ltx23VideoGenerator
from app.utils.ffmpeg import FFmpeg


def build_video_generator(settings: Settings, ffmpeg: FFmpeg) -> VideoGenerator:
    if settings.generator_backend == "fallback":
        return FallbackVideoGenerator(settings, ffmpeg)
    return Ltx23VideoGenerator(settings, ffmpeg)
