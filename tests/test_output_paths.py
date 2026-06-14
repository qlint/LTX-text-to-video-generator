from pathlib import Path

import pytest

from app.core.config import Settings
from app.services.output_paths import OutputPathError, OutputPathResolver


def test_resolves_numeric_subdirectory_and_prefix(tmp_path: Path) -> None:
    settings = Settings(output_allowed_roots=str(tmp_path))
    resolver = OutputPathResolver(settings)

    directory = resolver.resolve_directory(str(tmp_path), 12)
    output = resolver.build_output_path(directory, 3, 99)

    assert directory == tmp_path / "12"
    assert output.name == "99-3.mp4"


def test_rejects_directory_outside_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    disallowed = tmp_path / "disallowed"
    settings = Settings(output_allowed_roots=str(allowed))
    resolver = OutputPathResolver(settings)

    with pytest.raises(OutputPathError):
        resolver.resolve_directory(str(disallowed), None)


def test_sanitizes_prefix_without_changing_index(tmp_path: Path) -> None:
    settings = Settings(output_allowed_roots=str(tmp_path))
    resolver = OutputPathResolver(settings)

    directory = resolver.resolve_directory(str(tmp_path), "")
    output = resolver.build_output_path(directory, 0, "My Prefix/Bad")

    assert output.name == "My_Prefix_Bad-0.mp4"

