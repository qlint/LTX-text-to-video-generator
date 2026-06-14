from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoGenerationSpec:
    prompt: str
    output_path: Path
    seed: int
    width: int
    height: int
    num_frames: int
    frame_rate: int


class VideoGenerator:
    """Interface implemented by all model backends."""

    def generate(self, spec: VideoGenerationSpec) -> None:
        raise NotImplementedError

