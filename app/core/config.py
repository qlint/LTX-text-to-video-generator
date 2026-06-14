from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Self Hosted Text-to-Video Generator"
    environment: str = "production"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"

    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    job_ttl_seconds: int = 7 * 24 * 60 * 60

    generator_backend: Literal["ltx23", "fallback"] = "ltx23"
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    command_timeout_seconds: int = 0

    output_allowed_roots: str = "/var/www/html,/app/output"
    output_root_guard_enabled: bool = True
    output_file_index_base: int = 0
    output_extension: str = "mp4"
    overwrite_outputs: bool = True

    target_landscape_width: int = 1920
    target_landscape_height: int = 1080
    target_portrait_width: int = 1080
    target_portrait_height: int = 1920
    target_fps: int = 30
    use_motion_interpolation: bool = False

    ltx_python: str = "python"
    ltx_repo_path: str = "/opt/LTX-2"
    ltx_pipeline: Literal["hq", "standard", "distilled"] = "hq"
    ltx_checkpoint_path: str = "/models/LTX-2.3/ltx-2.3-22b-dev.safetensors"
    ltx_distilled_checkpoint_path: str = "/models/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
    ltx_distilled_lora_path: str = "/models/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    ltx_spatial_upsampler_path: str = "/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    ltx_gemma_root: str = "/models/gemma-3-12b-it-qat-q4_0-unquantized"
    ltx_quantization: str | None = "fp8-cast"
    ltx_offload_mode: Literal["none", "cpu", "disk"] = "none"
    ltx_max_batch_size: int = 1
    ltx_num_inference_steps: int = 15
    ltx_model_frame_rate: int = 48
    ltx_landscape_width: int = 1920
    ltx_landscape_height: int = 1088
    ltx_portrait_width: int = 1088
    ltx_portrait_height: int = 1920
    ltx_enhance_prompt: bool = True
    ltx_lora_paths: str = ""
    ltx_distilled_lora_strength: float = 0.8
    ltx_hq_distilled_lora_strength_stage_1: float = 0.25
    ltx_hq_distilled_lora_strength_stage_2: float = 0.5
    ltx_negative_prompt: str = Field(
        default=(
            "blurry, out of focus, overexposed, underexposed, low contrast, "
            "washed out colors, excessive noise, poor lighting, flickering, "
            "distorted proportions, deformed hands, incorrect anatomy, "
            "unreadable text, background clutter, harsh shadows, color banding, "
            "cartoonish rendering, CGI look, uncanny valley effect, mismatched "
            "lip sync, silent or muted audio, distorted voice, off-sync audio, "
            "jittery movement, unnatural transitions, AI artifacts"
        )
    )

    minimum_seconds_per_video: int = 6
    seconds_per_12_prompt_words: int = 1
    max_ltx_segment_seconds: int = 20
    max_prompt_words_per_segment: int = 145
    max_auto_segments: int = 0
    fallback_duration_seconds: int = 6

    @field_validator("api_prefix")
    @classmethod
    def normalize_api_prefix(cls, value: str) -> str:
        return "/" + value.strip("/")

    @field_validator("output_extension")
    @classmethod
    def normalize_extension(cls, value: str) -> str:
        return value.strip(".").lower()

    @property
    def allowed_output_roots(self) -> list[Path]:
        return [
            Path(part.strip()).expanduser().resolve(strict=False)
            for part in self.output_allowed_roots.split(",")
            if part.strip()
        ]

    @property
    def ltx_pythonpath_entries(self) -> list[str]:
        repo = Path(self.ltx_repo_path)
        return [
            str(repo / "packages" / "ltx-core" / "src"),
            str(repo / "packages" / "ltx-pipelines" / "src"),
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()

