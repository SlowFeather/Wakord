"""训练循环：BCE 损失，按验证集准确率保存最优权重。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .._logging import get_logger
from ..config import Config
from .dataset import Dataset
from .model import WakeWordModel

logger = get_logger(__name__)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_classifier(cfg: Config, data: Dataset) -> tuple[WakeWordModel, float]:
    """训练并返回 (载入最优权重的模型, 最优验证准确率)。"""
    device = _device()
    tf = cfg.train
    logger.info("训练设备: %s", device)

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
    best_acc = 0.0

    logger.info("开始训练 %d 轮...", tf.epochs)
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
            preds = (model(Xva) > 0.5).float()
            acc = (preds == yva).float().mean().item()

        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), fs.best_ckpt)

        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d  val_acc=%.2f%%", epoch + 1, tf.epochs, acc * 100)

    logger.info("最佳验证准确率: %.2f%%  权重已保存: %s", best_acc * 100, fs.best_ckpt)

    model.load_state_dict(torch.load(fs.best_ckpt, map_location=device))
    model.eval()
    return model, best_acc
