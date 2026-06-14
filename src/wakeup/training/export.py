"""导出模型：PyTorch -> ONNX（可选简化）-> 可选 TensorFlow SavedModel。

服务端运行只需要 ONNX，TensorFlow 链路是可选的（需 requirements-export.txt）。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from .._logging import get_logger
from ..config import Config
from .model import WakeWordModel

logger = get_logger(__name__)


def export_onnx(cfg: Config, model: WakeWordModel, *, simplify: bool = True) -> Path:
    """导出 ONNX，并把成品复制到部署路径 ``paths.model_path``。"""
    fs = cfg.fs
    fs.ensure_dirs()
    onnx_path = fs.onnx_model
    tf = cfg.train

    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(1, tf.target_frames, tf.embedding_dim).to(device)

    logger.info("导出 ONNX -> %s", onnx_path)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )

    if simplify:
        _try_simplify(onnx_path)

    # 复制到部署位置，服务默认从这里加载
    deploy = fs.model_path
    deploy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx_path, deploy)
    logger.info("成品模型已就绪: %s", deploy)
    return deploy


def _try_simplify(onnx_path: Path) -> None:
    try:
        import onnx
        from onnxsim import simplify

        model = onnx.load(str(onnx_path))
        simplified, ok = simplify(model)
        if ok:
            onnx.save(simplified, str(onnx_path))
            logger.info("ONNX 简化完成")
        else:
            logger.warning("ONNX 简化未通过校验，保留原始模型")
    except ImportError:
        logger.info("未安装 onnxsim，跳过简化（pip install onnxsim）")
    except Exception as exc:
        logger.warning("ONNX 简化失败，保留原始模型: %s", exc)


def export_tensorflow(cfg: Config) -> Path | None:
    """ONNX -> TensorFlow SavedModel（可选）。需安装 onnx2tf。"""
    fs = cfg.fs
    onnx_path = fs.onnx_model
    tf_path = fs.tf_model
    if not onnx_path.exists():
        raise FileNotFoundError(f"未找到 ONNX 模型: {onnx_path}，请先导出 ONNX")

    try:
        import onnx2tf
    except ImportError:
        logger.warning(
            "未安装 onnx2tf，跳过 TF 导出（pip install -r requirements-export.txt）"
        )
        return None

    logger.info("转换 ONNX -> TensorFlow SavedModel -> %s", tf_path)
    onnx2tf.convert(input_onnx_file_path=str(onnx_path), output_folder_path=str(tf_path))
    logger.info("TensorFlow 模型输出: %s", tf_path)
    return tf_path
