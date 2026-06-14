from __future__ import annotations

import json
import os
from pathlib import Path

from app.core.config import Settings
from app.inference.base import VideoGenerationSpec, VideoGenerator
from app.utils.ffmpeg import FFmpeg

PIPELINE_MODULES = {
    "hq": "ltx_pipelines.ti2vid_two_stages_hq",
    "standard": "ltx_pipelines.ti2vid_two_stages",
    "distilled": "ltx_pipelines.distilled",
}


class Ltx23VideoGenerator(VideoGenerator):
    """Subprocess adapter around the official Lightricks LTX-2.3 pipelines."""

    def __init__(self, settings: Settings, ffmpeg: FFmpeg) -> None:
        self.settings = settings
        self.ffmpeg = ffmpeg

    def generate(self, spec: VideoGenerationSpec) -> None:
        self._validate_runtime()
        command = self._command(spec)
        env = self._environment()
        self.ffmpeg.run(command, cwd=self.settings.ltx_repo_path, env=env)

    def _command(self, spec: VideoGenerationSpec) -> list[str]:
        pipeline_module = PIPELINE_MODULES[self.settings.ltx_pipeline]
        command = [self.settings.ltx_python, "-m", pipeline_module]

        if self.settings.ltx_pipeline == "distilled":
            command.extend(
                [
                    "--distilled-checkpoint-path",
                    self.settings.ltx_distilled_checkpoint_path,
                ]
            )
        else:
            command.extend(["--checkpoint-path", self.settings.ltx_checkpoint_path])

        command.extend(
            [
                "--gemma-root",
                self.settings.ltx_gemma_root,
                "--prompt",
                spec.prompt,
                "--output-path",
                str(spec.output_path),
                "--seed",
                str(spec.seed),
                "--height",
                str(spec.height),
                "--width",
                str(spec.width),
                "--num-frames",
                str(spec.num_frames),
                "--frame-rate",
                str(spec.frame_rate),
                "--spatial-upsampler-path",
                self.settings.ltx_spatial_upsampler_path,
                "--offload",
                self.settings.ltx_offload_mode,
                "--max-batch-size",
                str(self.settings.ltx_max_batch_size),
            ]
        )

        if self.settings.ltx_pipeline != "distilled":
            command.extend(
                [
                    "--negative-prompt",
                    self.settings.ltx_negative_prompt,
                    "--num-inference-steps",
                    str(self.settings.ltx_num_inference_steps),
                    "--distilled-lora",
                    self.settings.ltx_distilled_lora_path,
                    str(self.settings.ltx_distilled_lora_strength),
                ]
            )

        if self.settings.ltx_pipeline == "hq":
            command.extend(
                [
                    "--distilled-lora-strength-stage-1",
                    str(self.settings.ltx_hq_distilled_lora_strength_stage_1),
                    "--distilled-lora-strength-stage-2",
                    str(self.settings.ltx_hq_distilled_lora_strength_stage_2),
                ]
            )

        if self.settings.ltx_enhance_prompt:
            command.append("--enhance-prompt")

        if self.settings.ltx_quantization:
            command.extend(["--quantization", self.settings.ltx_quantization])

        for lora_path, strength in self._loras():
            command.extend(["--lora", lora_path, strength])

        return command

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        entries = [*self.settings.ltx_pythonpath_entries]
        if pythonpath:
            entries.append(pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(entries)
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        return env

    def _validate_runtime(self) -> None:
        repo = Path(self.settings.ltx_repo_path)
        if not repo.exists():
            raise RuntimeError(
                f"LTX repository not found at {repo}. Clone Lightricks/LTX-2 or set LTX_REPO_PATH."
            )

        required_directories = [
            Path(self.settings.ltx_gemma_root),
        ]
        required_files = [
            Path(self.settings.ltx_spatial_upsampler_path),
        ]
        if self.settings.ltx_pipeline == "distilled":
            required_files.append(Path(self.settings.ltx_distilled_checkpoint_path))
        else:
            required_files.extend(
                [
                    Path(self.settings.ltx_checkpoint_path),
                    Path(self.settings.ltx_distilled_lora_path),
                ]
            )

        missing = [
            str(path)
            for path in [*required_directories, *required_files]
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                "LTX model assets are missing. Download or mount: " + ", ".join(missing)
            )

        empty_directories = [
            str(path)
            for path in required_directories
            if path.is_dir() and not any(path.iterdir())
        ]
        if empty_directories:
            raise RuntimeError(
                "LTX model asset directories are empty. Re-download: "
                + ", ".join(empty_directories)
            )

        gemma_errors = self._validate_gemma_directory(Path(self.settings.ltx_gemma_root))
        if gemma_errors:
            raise RuntimeError(
                "Gemma text encoder assets are incomplete. Accept the Gemma license, "
                "authenticate Hugging Face CLI if needed, then re-run "
                "./scripts/download-ltx23-models.sh. Missing: "
                + ", ".join(gemma_errors)
            )

        invalid = []
        for path in required_files:
            if path.suffix == ".safetensors":
                try:
                    self._validate_safetensors_file(path)
                except ValueError as exc:
                    invalid.append(f"{path} ({exc})")

        if invalid:
            raise RuntimeError(
                "LTX model assets are invalid or incomplete. Delete and re-download: "
                + ", ".join(invalid)
            )

    def _validate_safetensors_file(self, path: Path) -> None:
        size = path.stat().st_size
        if size <= 8:
            raise ValueError("file is too small")

        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            header_end = 8 + header_size
            if header_size <= 0 or header_end > size:
                raise ValueError("invalid safetensors header length")
            header = json.loads(handle.read(header_size).decode("utf-8"))

        data_size = size - header_end
        offsets = [
            value.get("data_offsets")
            for key, value in header.items()
            if key != "__metadata__" and isinstance(value, dict)
        ]
        offsets = [offset for offset in offsets if offset is not None]
        if not offsets:
            raise ValueError("no tensor data offsets found")

        max_end = 0
        for offset in offsets:
            if (
                not isinstance(offset, list)
                or len(offset) != 2
                or offset[0] < 0
                or offset[1] < offset[0]
                or offset[1] > data_size
            ):
                raise ValueError("invalid tensor data offsets")
            max_end = max(max_end, offset[1])

        if max_end != data_size:
            raise ValueError("safetensors data is incomplete or has trailing bytes")

    def _validate_gemma_directory(self, path: Path) -> list[str]:
        missing = []
        required_files = ["config.json", "model.safetensors.index.json"]
        for filename in required_files:
            if not (path / filename).is_file():
                missing.append(str(path / filename))

        if not list(path.glob("model-*.safetensors")):
            missing.append(str(path / "model-*.safetensors"))

        return missing

    def _loras(self) -> list[tuple[str, str]]:
        if not self.settings.ltx_lora_paths.strip():
            return []
        parsed = []
        for item in self.settings.ltx_lora_paths.split(","):
            value = item.strip()
            if not value:
                continue
            if ":" in value:
                path, strength = value.rsplit(":", 1)
            else:
                path, strength = value, "1.0"
            parsed.append((path, strength))
        return parsed
