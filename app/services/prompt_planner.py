from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings

PORTRAIT_HINTS = (
    "portrait",
    "vertical",
    "9:16",
    "tiktok",
    "reel",
    "reels",
    "shorts",
    "phone screen",
    "mobile video",
)


@dataclass(frozen=True)
class SegmentPlan:
    prompt: str
    seconds: int
    num_frames: int
    seed: int


@dataclass(frozen=True)
class PromptPlan:
    prompt_index: int
    orientation: Literal["landscape", "portrait"]
    model_width: int
    model_height: int
    target_width: int
    target_height: int
    segments: list[SegmentPlan]


class PromptPlanner:
    """Build consistent, model-ready prompts and duration chunks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def plan_batch(self, prompts: list[str], scene_seed: int) -> list[PromptPlan]:
        shared_context = self._shared_context(prompts)
        return [
            self.plan_prompt(index, prompt, scene_seed, shared_context)
            for index, prompt in enumerate(prompts)
        ]

    def plan_prompt(
        self,
        prompt_index: int,
        prompt: str,
        scene_seed: int,
        shared_context: str,
    ) -> PromptPlan:
        orientation = self._orientation(prompt)
        model_width, model_height, target_width, target_height = self._dimensions(orientation)
        chunks = self._prompt_chunks(prompt)
        total_seconds = self._estimate_seconds(prompt)
        segment_count = max(
            len(chunks),
            math.ceil(total_seconds / self.settings.max_ltx_segment_seconds),
        )
        if self.settings.max_auto_segments > 0:
            segment_count = min(segment_count, self.settings.max_auto_segments)
        chunks = self._fit_chunk_count(prompt, chunks, segment_count)
        seconds_per_segment = max(
            self.settings.minimum_seconds_per_video,
            math.ceil(total_seconds / len(chunks)),
        )

        segments = []
        for segment_index, chunk in enumerate(chunks):
            segment_prompt = self._segment_prompt(
                chunk=chunk,
                shared_context=shared_context,
                segment_index=segment_index,
                segment_total=len(chunks),
            )
            seed = scene_seed + (prompt_index * 1000) + segment_index
            num_frames = self._num_frames(seconds_per_segment)
            segments.append(
                SegmentPlan(
                    prompt=segment_prompt,
                    seconds=seconds_per_segment,
                    num_frames=num_frames,
                    seed=seed,
                )
            )

        return PromptPlan(
            prompt_index=prompt_index,
            orientation=orientation,
            model_width=model_width,
            model_height=model_height,
            target_width=target_width,
            target_height=target_height,
            segments=segments,
        )

    def _shared_context(self, prompts: list[str]) -> str:
        joined = " ".join(prompts)
        scene_words = " ".join(self._words(joined)[:80])
        return (
            "Maintain one continuous visual universe across every clip in this batch: "
            "same location, subject identities, wardrobe, production design, lens language, "
            "time of day, lighting continuity, color grade, and environmental sound unless "
            "a prompt explicitly changes it. Favor realism, professional exposure, stable "
            "camera motion, clean composition, and an uncluttered background when it suits "
            f"the scene. Batch scene context: {scene_words}."
        )

    def _segment_prompt(
        self,
        chunk: str,
        shared_context: str,
        segment_index: int,
        segment_total: int,
    ) -> str:
        continuity = ""
        if segment_total > 1:
            continuity = (
                f"Segment {segment_index + 1} of {segment_total}; continue the same "
                "scene naturally with no visual reset. "
            )
        return (
            f"{continuity}{chunk.strip()} {shared_context} Include synchronized ambient "
            "sound, foley, or speech only when the prompt implies it."
        )

    def _estimate_seconds(self, prompt: str) -> int:
        word_count = len(self._words(prompt))
        seconds = max(
            self.settings.minimum_seconds_per_video,
            math.ceil(word_count / 12) * self.settings.seconds_per_12_prompt_words,
        )
        return seconds

    def _prompt_chunks(self, prompt: str) -> list[str]:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", prompt) if part.strip()]
        if not sentences:
            sentences = [prompt]

        chunks: list[str] = []
        current: list[str] = []
        current_words = 0
        limit = self.settings.max_prompt_words_per_segment

        for sentence in sentences:
            count = len(self._words(sentence))
            if current and current_words + count > limit:
                chunks.append(" ".join(current))
                current = [sentence]
                current_words = count
            else:
                current.append(sentence)
                current_words += count

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _fit_chunk_count(self, prompt: str, chunks: list[str], segment_count: int) -> list[str]:
        if len(chunks) >= segment_count:
            return chunks[:segment_count]
        words = self._words(prompt)
        if not words:
            return chunks

        words_per_chunk = max(1, math.ceil(len(words) / segment_count))
        expanded = [
            " ".join(words[index : index + words_per_chunk])
            for index in range(0, len(words), words_per_chunk)
        ]
        return expanded[:segment_count]

    def _num_frames(self, seconds: int) -> int:
        raw_frames = max(1, round(seconds * self.settings.ltx_model_frame_rate))
        k = max(1, math.ceil(raw_frames / 8))
        return (8 * k) + 1

    def _orientation(self, prompt: str) -> Literal["landscape", "portrait"]:
        lowered = prompt.lower()
        if any(hint in lowered for hint in PORTRAIT_HINTS):
            return "portrait"
        return "landscape"

    def _dimensions(
        self,
        orientation: Literal["landscape", "portrait"],
    ) -> tuple[int, int, int, int]:
        if orientation == "portrait":
            return (
                self.settings.ltx_portrait_width,
                self.settings.ltx_portrait_height,
                self.settings.target_portrait_width,
                self.settings.target_portrait_height,
            )
        return (
            self.settings.ltx_landscape_width,
            self.settings.ltx_landscape_height,
            self.settings.target_landscape_width,
            self.settings.target_landscape_height,
        )

    @staticmethod
    def _words(text: str) -> list[str]:
        return re.findall(r"\b[\w'-]+\b", text)
