from __future__ import annotations

from app.core.config import Settings
from app.inference.base import VideoGenerationSpec, VideoGenerator
from app.utils.ffmpeg import FFmpeg


class FallbackVideoGenerator(VideoGenerator):
    """Tiny local renderer used only for health checks and development tests."""

    def __init__(self, settings: Settings, ffmpeg: FFmpeg) -> None:
        self.settings = settings
        self.ffmpeg = ffmpeg

    def generate(self, spec: VideoGenerationSpec) -> None:
        self.ffmpeg.generate_fallback_video(
            output_path=spec.output_path,
            duration_seconds=self.settings.fallback_duration_seconds,
            width=spec.width,
            height=spec.height,
        )

