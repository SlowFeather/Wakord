"""Ensure openWakeWord feature/VAD models are available in a project cache."""

from __future__ import annotations

import pathlib
import shutil

from .._logging import get_logger
from ..config import Config, PathsConfig
from ._download import download

logger = get_logger(__name__)

_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
_FEATURE_MODELS = ("melspectrogram.onnx", "embedding_model.onnx")
_VAD_MODELS = ("silero_vad.onnx",)


def package_models_dir() -> pathlib.Path:
    """Return openWakeWord's package resources/models directory."""
    import openwakeword

    return pathlib.Path(openwakeword.__file__).parent / "resources" / "models"


def oww_models_dir(cfg: Config | None = None) -> pathlib.Path:
    """Return the project cache directory for openWakeWord support models."""
    if cfg is not None:
        return cfg.fs.oww_models_dir
    return pathlib.Path(PathsConfig().oww_models_dir)


def _mirror_to_package(cache_file: pathlib.Path) -> None:
    """Best-effort compatibility mirror for openWakeWord APIs that use package paths."""
    try:
        package_dir = package_models_dir()
        package_dir.mkdir(parents=True, exist_ok=True)
        dest = package_dir / cache_file.name
        if not dest.exists() or dest.stat().st_size != cache_file.stat().st_size:
            shutil.copyfile(cache_file, dest)
    except Exception as exc:
        logger.warning(
            "Could not mirror %s into openWakeWord package resources; cache remains available: %s",
            cache_file.name,
            exc,
        )


def ensure_feature_models(cfg: Config | None = None, include_vad: bool = False) -> pathlib.Path:
    """Ensure feature extraction models exist in the project cache and return that path."""
    target = oww_models_dir(cfg)
    target.mkdir(parents=True, exist_ok=True)

    names = list(_FEATURE_MODELS) + (list(_VAD_MODELS) if include_vad else [])
    for name in names:
        dest = target / name
        if not dest.exists() or dest.stat().st_size <= 0:
            logger.info("Fetching openWakeWord support model: %s", name)
            download(f"{_RELEASE}/{name}", dest, desc=name)
        _mirror_to_package(dest)
    return target
