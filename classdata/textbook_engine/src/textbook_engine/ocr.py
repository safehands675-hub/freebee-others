from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

from .core import ContentEngineError


def _windows_user_environment() -> dict[str, str]:
    if sys.platform != "win32":
        return {}

    import winreg

    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment")
    except FileNotFoundError:
        return {}

    settings: dict[str, str] = {}
    with key:
        for name in ("TESSERACT_CMD", "TESSDATA_PREFIX"):
            try:
                value, _ = winreg.QueryValueEx(key, name)
            except FileNotFoundError:
                continue
            if isinstance(value, str) and value.strip():
                settings[name] = value.strip()
    return settings


def _load_tesseract_settings() -> dict[str, str]:
    settings = _windows_user_environment()
    for name in ("TESSERACT_CMD", "TESSDATA_PREFIX"):
        process_value = os.environ.get(name)
        if process_value:
            settings[name] = process_value
        elif settings.get(name):
            os.environ[name] = settings[name]
    return settings


def resolve_tesseract_command() -> str:
    settings = _load_tesseract_settings()
    command = settings.get("TESSERACT_CMD", "tesseract")
    resolved = shutil.which(command)
    if resolved is None:
        raise ContentEngineError(
            "Tesseract OCR was not found. Install Tesseract OCR, add it to PATH, "
            "or set TESSERACT_CMD to its executable path."
        )
    return resolved


def resolve_tesseract_languages(executable: str) -> set[str]:
    try:
        result = subprocess.run(
            [executable, "--list-langs"],
            capture_output=True,
            check=False,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise ContentEngineError("Tesseract language detection timed out.") from exc
    except OSError as exc:
        raise ContentEngineError(f"Could not query Tesseract languages: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ContentEngineError(
            f"Could not query Tesseract languages: {detail or 'unknown OCR error'}"
        )
    output = result.stdout.decode("utf-8", errors="replace")
    return {
        match.group(1)
        for line in output.splitlines()
        if (match := re.fullmatch(r"\s*([A-Za-z0-9_]+)\s*", line))
    }
