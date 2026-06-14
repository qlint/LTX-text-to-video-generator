from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.core.config import Settings

SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class OutputPathError(ValueError):
    """Raised when a requested output location is unsafe or invalid."""


class OutputPathResolver:
    """Resolve and validate target directories and filenames."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def resolve_directory(
        self,
        asset_directory: str,
        asset_subdirectory: str | int | float | None,
    ) -> Path:
        base = Path(asset_directory).expanduser()
        if not base.is_absolute():
            raise OutputPathError("asset_directory must be an absolute path")

        subdirectory = self._string_or_none(asset_subdirectory)
        if subdirectory:
            sub_path = Path(subdirectory)
            if sub_path.is_absolute():
                raise OutputPathError("asset_subdirectory must be relative")
            candidate = base / sub_path
        else:
            candidate = base

        resolved = candidate.resolve(strict=False)
        self._assert_allowed(resolved)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def build_output_path(
        self,
        directory: Path,
        prompt_index: int,
        naming_prefix: str | int | float | None,
    ) -> Path:
        displayed_index = prompt_index + self.settings.output_file_index_base
        prefix = self.safe_filename_part(naming_prefix)
        if prefix:
            filename = f"{prefix}-{displayed_index}.{self.settings.output_extension}"
        else:
            filename = f"{displayed_index}.{self.settings.output_extension}"
        return directory / filename

    def safe_filename_part(self, value: str | int | float | None) -> str | None:
        raw = self._string_or_none(value)
        if raw is None:
            return None
        cleaned = SAFE_FILENAME_PATTERN.sub("_", raw.strip()).strip("._-")
        return cleaned or None

    def _assert_allowed(self, path: Path) -> None:
        if not self.settings.output_root_guard_enabled:
            return

        roots = self.settings.allowed_output_roots
        if not roots:
            raise OutputPathError("no output_allowed_roots configured")

        for root in roots:
            try:
                path.relative_to(root)
                return
            except ValueError:
                continue

        allowed = ", ".join(str(root) for root in roots)
        raise OutputPathError(f"output path is outside allowed roots: {allowed}")

    @staticmethod
    def _string_or_none(value: Any) -> str | None:
        if value is None:
            return None
        rendered = str(value).strip()
        return rendered or None
