"""Helpers for optional LHD shot metadata."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SHOTINFO_DIR_ENV = "LHD_SHOTINFO_DIR"


def get_shot_info(
    shot: int,
    *,
    executable: str = "getShotInfo.exe",
    cwd: str | Path | None = None,
) -> dict[str, float | int]:
    """Return shot metadata from the locally installed ``getShotInfo`` tool."""

    working_dir = Path(cwd) if cwd is not None else _env_working_dir()
    command = [executable, str(int(shot))]

    try:
        completed = subprocess.run(
            command,
            cwd=working_dir,
            capture_output=True,
            check=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "getShotInfo executable was not found. Install/configure the LHD tools "
            "outside this package or pass an explicit executable path."
        ) from exc

    return parse_shot_info_output(completed.stdout, shot=shot)


def parse_shot_info_output(output: str, *, shot: int) -> dict[str, float | int]:
    """Parse the comma-separated output emitted by ``getShotInfo``."""

    values = [float(value) for value in output.strip().split(",") if value.strip()]
    keys = ("Bt", "Rax", "Bq", "Gamma", "ExpDate")
    if len(values) < len(keys):
        raise ValueError(f"Unexpected getShotInfo output: {output!r}")

    result: dict[str, float | int] = dict(zip(keys, values, strict=False))
    result["ExpDate"] = int(result["ExpDate"])
    result["shot"] = int(shot)
    return result


def _env_working_dir() -> Path | None:
    value = os.environ.get(SHOTINFO_DIR_ENV)
    return Path(value).expanduser() if value else None
