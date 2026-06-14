"""唤醒词分类器：在 openWakeWord 的 (T=16, D=96) 特征上做二分类的小型 CNN。

输入: (B, 16, 96)  ——  16 帧、每帧 96 维 embedding
输出: (B, 1)       ——  sigmoid 唤醒概率

该输入/输出约定与 openWakeWord 运行时对自定义模型的要求一致，
因此导出的 onnx 可以直接被 openWakeWord 的 Model 加载。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WakeWordModel(nn.Module):
    def __init__(self, target_frames: int = 16, embedding_dim: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 两次 2x2 池化后展平维度：64 * (T/4) * (D/4)
        flat = 64 * (target_frames // 4) * (embedding_dim // 4)
        self.fc1 = nn.Linear(flat, 64)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, T, D)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return torch.sigmoid(self.fc2(x))
