"""训练循环：BCE 损失，按验证集准确率保存最优权重。"""

from __future__ import annotations

import json

import numpy as np

from .._logging import get_logger
from ..config import Config
from .dataset import Dataset

logger = get_logger(__name__)


def classification_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute binary wake-word metrics for one threshold."""
    pred = scores >= threshold
    truth = labels.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def threshold_scan(scores: np.ndarray, labels: np.ndarray) -> tuple[list[dict], dict]:
    """Scan practical thresholds and choose the strongest F1 operating point."""
    scan = [classification_metrics(scores, labels, t) for t in np.linspace(0.1, 0.9, 17)]
    best = max(scan, key=lambda m: (m["f1"], m["recall"], -m["false_positive_rate"]))
    return scan, best


def _resolve_device(pref: str) -> str:
    """把 ``auto|cuda|cpu`` 解析成实际设备，并在请求 GPU 不可用时给出明确提示。"""
    import torch

    pref = (pref or "auto").lower()
    has_cuda = torch.cuda.is_available()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if not has_cuda:
            raise RuntimeError(
                "配置要求用 CUDA，但 torch.cuda.is_available()=False。\n"
                "多半是装了 CPU 版 torch（torch.__version__ 带 +cpu）。\n"
                "请装 CUDA 版：pip uninstall -y torch && "
                "pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )
        return "cuda"
    # auto
    return "cuda" if has_cuda else "cpu"


def train_classifier(cfg: Config, data: Dataset):
    """Train and return (best-weight model, best validation F1 at the recommended threshold)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from .model import WakeWordModel

    tf = cfg.train
    device = _resolve_device(tf.device)
    if device == "cuda":
        logger.info("Training device: cuda (%s)", torch.cuda.get_device_name(0))
    else:
        logger.info("Training device: cpu (install a CUDA torch build to use GPU)")

    torch.manual_seed(tf.seed)

    Xtr = torch.from_numpy(data.X_train).to(device)
    ytr = torch.from_numpy(data.y_train).unsqueeze(1).to(device)
    Xva = torch.from_numpy(data.X_val).to(device)
    yva = torch.from_numpy(data.y_val).unsqueeze(1).to(device)

    model = WakeWordModel(tf.target_frames, tf.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=tf.learning_rate)
    criterion = nn.BCELoss()

    fs = cfg.fs
    fs.ensure_dirs()
    best_f1 = -1.0
    best_metrics: dict | None = None

    logger.info("Starting training for %d epochs...", tf.epochs)
    for epoch in range(tf.epochs):
        model.train()
        for i in range(0, len(Xtr), tf.batch_size):
            bx = Xtr[i : i + tf.batch_size]
            by = ytr[i : i + tf.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scores = model(Xva).detach().cpu().numpy().reshape(-1)
        metrics = classification_metrics(val_scores, data.y_val, threshold=0.5)

        if metrics["f1"] >= best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics
            torch.save(model.state_dict(), fs.best_ckpt)

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Epoch %d/%d  val_f1=%.3f precision=%.3f recall=%.3f fpr=%.3f",
                epoch + 1, tf.epochs, metrics["f1"], metrics["precision"],
                metrics["recall"], metrics["false_positive_rate"],
            )

    logger.info("Best validation F1: %.3f  checkpoint saved: %s", best_f1, fs.best_ckpt)

    model.load_state_dict(torch.load(fs.best_ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        scores = model(Xva).detach().cpu().numpy().reshape(-1)
    scan, recommended = threshold_scan(scores, data.y_val)
    metrics_payload = {
        "selection_metric": "f1@0.5",
        "best_epoch_metrics": best_metrics,
        "recommended_threshold": recommended["threshold"],
        "recommended_metrics": recommended,
        "threshold_scan": scan,
        "class_balance": {
            "train_positive": int(data.y_train.sum()),
            "train_negative": int((data.y_train == 0).sum()),
            "val_positive": int(data.y_val.sum()),
            "val_negative": int((data.y_val == 0).sum()),
        },
    }
    metrics_path = fs.model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Recommended threshold %.2f: f1=%.3f precision=%.3f recall=%.3f fpr=%.3f; metrics written to %s",
        recommended["threshold"], recommended["f1"], recommended["precision"],
        recommended["recall"], recommended["false_positive_rate"], metrics_path,
    )
    return model, recommended["f1"]
