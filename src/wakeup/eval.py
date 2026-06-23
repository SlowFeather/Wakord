"""Offline evaluation helpers for real wake-word acceptance audio."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ._logging import get_logger
from .config import Config
from .data.features import _finalize_clip, _load_audio
from .service.detector import WakeWordDetector
from .training.trainer import classification_metrics, threshold_scan

logger = get_logger(__name__)

_AUDIO_GLOBS = ("*.wav", "*.mp3", "*.flac", "*.ogg")


@dataclass
class ClipResult:
    path: str
    label: int
    score: float
    triggered: bool
    trigger_count: int


def iter_audio_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    files: list[Path] = []
    for pattern in _AUDIO_GLOBS:
        files.extend(folder.rglob(pattern))
    return sorted(files)


def _score_clip(cfg: Config, detector: WakeWordDetector, path: Path) -> tuple[float, int]:
    audio = _load_audio(str(path), cfg.service.sample_rate)
    if audio is None:
        raise RuntimeError(f"could not read audio: {path}")
    frame_len = cfg.service.frame_samples
    target_len = max(frame_len, int(np.ceil(len(audio) / frame_len) * frame_len))
    pcm = _finalize_clip(audio, target_len)
    detector.reset()
    peak = 0.0
    hits = 0
    for start in range(0, len(pcm), frame_len):
        frame = pcm[start : start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)), mode="constant")
        event = detector.process(frame)
        peak = max(peak, detector.last_active_score)
        if event is not None:
            hits += 1
    return peak, hits


def evaluate_audio_dirs(
    cfg: Config,
    positive_dir: Path,
    negative_dir: Path,
    *,
    threshold: float | None = None,
    out_json: Path | None = None,
    out_csv: Path | None = None,
) -> dict:
    """Score real positive/negative audio directories and write optional reports."""
    threshold = cfg.service.threshold if threshold is None else threshold
    pos_files = iter_audio_files(positive_dir)
    neg_files = iter_audio_files(negative_dir)
    if not pos_files and not neg_files:
        raise RuntimeError("no audio files found in positive or negative evaluation directories")

    detector = WakeWordDetector(cfg)
    rows: list[ClipResult] = []
    for label, files in ((1, pos_files), (0, neg_files)):
        for path in files:
            score, hits = _score_clip(cfg, detector, path)
            rows.append(
                ClipResult(
                    path=str(path),
                    label=label,
                    score=score,
                    triggered=score >= threshold,
                    trigger_count=hits,
                )
            )
            logger.info("eval %s label=%d score=%.3f hits=%d", path, label, score, hits)

    labels = np.array([r.label for r in rows], dtype=np.float32)
    scores = np.array([r.score for r in rows], dtype=np.float32)
    metrics = classification_metrics(scores, labels, threshold=threshold)
    scan, recommended = threshold_scan(scores, labels) if len(set(labels.tolist())) > 1 else ([], metrics)
    payload = {
        "threshold": threshold,
        "positive_dir": str(positive_dir),
        "negative_dir": str(negative_dir),
        "counts": {"positive": len(pos_files), "negative": len(neg_files), "total": len(rows)},
        "metrics": metrics,
        "recommended_threshold": recommended.get("threshold"),
        "recommended_metrics": recommended,
        "threshold_scan": scan,
        "clips": [asdict(r) for r in rows],
    }

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
    return payload


def default_eval_dirs(cfg: Config) -> tuple[Path, Path]:
    root = cfg.fs.data_dir / "eval"
    return root / "positive", root / "negative"
