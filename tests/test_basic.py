"""轻量单元测试（不依赖音频/模型，可在 CI 直接跑）。

    pip install pytest && pytest
"""

import numpy as np

from wakeup.config import load_config
from wakeup.training.dataset import fix_frames


def test_config_defaults():
    cfg = load_config()
    assert cfg.data.target_word == "小元"
    assert cfg.service.port == 8765
    assert cfg.fs.model_path.name == "xiaoyuan.onnx"


def test_fix_frames_pads_short():
    x = np.zeros((4, 8, 96), dtype=np.float32)
    out = fix_frames(x, 16)
    assert out.shape == (4, 16, 96)


def test_fix_frames_truncates_long():
    x = np.zeros((4, 32, 96), dtype=np.float32)
    out = fix_frames(x, 16)
    assert out.shape == (4, 16, 96)


def test_fix_frames_expands_2d():
    x = np.zeros((4, 96), dtype=np.float32)
    out = fix_frames(x, 16)
    assert out.shape == (4, 16, 96)


def test_protocol_roundtrip():
    from wakeup.service import protocol as p

    msg = {"type": "wake", "model": "小元", "score": 0.97}
    assert p.decode(p.encode(msg)) == msg
