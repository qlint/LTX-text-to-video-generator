from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from app.core.config import Settings
from app.inference.base import VideoGenerationSpec, VideoGenerator
from app.models.schemas import VideoGenerationRequest
from app.services.job_store import JobStore
from app.services.output_paths import OutputPathResolver
from app.services.prompt_planner import PromptPlan, PromptPlanner
from app.utils.ffmpeg import FFmpeg

logger = logging.getLogger(__name__)


class VideoOrchestrator:
    """Coordinate prompt planning, model execution, post-processing, and cleanup."""

    def __init__(
        self,
        settings: Settings,
        generator: VideoGenerator,
        ffmpeg: FFmpeg,
        resolver: OutputPathResolver,
        planner: PromptPlanner,
    ) -> None:
        self.settings = settings
        self.generator = generator
        self.ffmpeg = ffmpeg
        self.resolver = resolver
        self.planner = planner

    def generate_batch(
        self,
        job_id: str,
        request: VideoGenerationRequest,
        job_store: JobStore,
    ) -> list[Path]:
        output_dir = self.resolver.resolve_directory(
            request.asset_directory,
            request.asset_subdirectory,
        )
        scene_seed = int(job_id.replace("-", "")[:8], 16)
        plans = self.planner.plan_batch(request.prompts, scene_seed)

        final_paths: list[Path] = []
        for plan in plans:
            final_path = self.resolver.build_output_path(
                output_dir,
                plan.prompt_index,
                request.naming_prefix,
            )
            final_path = self._available_path(final_path)
            job_store.mark_prompt_running(job_id, plan.prompt_index, str(final_path))
            try:
                self._generate_prompt(plan, final_path)
                job_store.mark_prompt_completed(job_id, plan.prompt_index, str(final_path))
                final_paths.append(final_path)
            except Exception as exc:
                logger.exception("prompt generation failed", extra={"job_id": job_id})
                job_store.mark_prompt_failed(
                    job_id,
                    plan.prompt_index,
                    str(final_path),
                    str(exc),
                )
                raise

        return final_paths

    def _generate_prompt(self, plan: PromptPlan, final_path: Path) -> None:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".ttv-{plan.prompt_index}-",
            dir=final_path.parent,
        ) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            normalized_segments: list[Path] = []

            for segment_index, segment in enumerate(plan.segments):
                raw_path = temp_dir / f"raw-{segment_index:03d}.{self.settings.output_extension}"
                normalized_path = temp_dir / f"segment-{segment_index:03d}.{self.settings.output_extension}"
                spec = VideoGenerationSpec(
                    prompt=segment.prompt,
                    output_path=raw_path,
                    seed=segment.seed,
                    width=plan.model_width,
                    height=plan.model_height,
                    num_frames=segment.num_frames,
                    frame_rate=self.settings.ltx_model_frame_rate,
                )
                self.generator.generate(spec)
                self.ffmpeg.normalize(
                    input_path=raw_path,
                    output_path=normalized_path,
                    target_width=plan.target_width,
                    target_height=plan.target_height,
                )
                normalized_segments.append(normalized_path)

            final_temp = temp_dir / f"final.{self.settings.output_extension}"
            if len(normalized_segments) == 1:
                normalized_segments[0].replace(final_temp)
            else:
                self.ffmpeg.concat(
                    segment_paths=normalized_segments,
                    output_path=final_temp,
                    list_path=temp_dir / "segments.txt",
                )

            final_temp.replace(final_path)

    def _available_path(self, path: Path) -> Path:
        if self.settings.overwrite_outputs or not path.exists():
            return path

        counter = 1
        while True:
            candidate = path.with_name(f"{path.stem}-{counter}{path.suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

