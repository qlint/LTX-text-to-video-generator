from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from app.core.config import Settings

logger = logging.getLogger(__name__)


class CommandExecutionError(RuntimeError):
    """Raised when an external command fails."""


class FFmpeg:
    """FFmpeg wrapper for validation-friendly command execution."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def normalize(
        self,
        input_path: Path,
        output_path: Path,
        target_width: int,
        target_height: int,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vf = self._video_filter(target_width, target_height)
        command = [
            self.settings.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            vf,
            "-r",
            str(self.settings.target_fps),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self.run(command)

    def concat(self, segment_paths: list[Path], output_path: Path, list_path: Path) -> None:
        list_text = "".join(
            f"file '{str(path).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
            for path in segment_paths
        )
        list_path.write_text(list_text, encoding="utf-8")
        command = [
            self.settings.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        self.run(command)

    def generate_fallback_video(
        self,
        output_path: Path,
        duration_seconds: int,
        width: int,
        height: int,
    ) -> None:
        command = [
            self.settings.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={self.settings.target_fps}:duration={duration_seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={duration_seconds}",
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self.run(command)

    def run(self, command: list[str], cwd: str | None = None, env: dict[str, str] | None = None) -> None:
        logger.info("running command", extra={"command": command})
        timeout = self.settings.command_timeout_seconds or None
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise CommandExecutionError(
                "command failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )

    def _video_filter(self, target_width: int, target_height: int) -> str:
        scale_crop = (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
            f"crop={target_width}:{target_height}"
        )
        fps_filter = (
            f"minterpolate=fps={self.settings.target_fps}:mi_mode=mci"
            if self.settings.use_motion_interpolation
            else f"fps={self.settings.target_fps}"
        )
        return f"{scale_crop},{fps_filter}"

