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
        opset_version=18,
        verbose=False,  # torch2.x dynamo 导出器默认会 print 含 emoji 的进度，GBK 控制台会崩
    )

    # torch>=2.x 的 dynamo 导出器可能把权重存成外部 .data 旁文件，单拷 .onnx 会丢权重，
    # 导致部署模型加载失败。这里统一内联成单个自包含文件。
    _inline_external_data(onnx_path)

    if simplify:
        _try_simplify(onnx_path)

    # 复制到部署位置，服务默认从这里加载
    deploy = fs.model_path
    deploy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx_path, deploy)
    logger.info("成品模型已就绪: %s", deploy)
    return deploy


def export_from_checkpoint(cfg: Config, *, simplify: bool = True) -> Path:
    """从已训练的 ``best.pth`` 重新导出 ONNX，无需重新训练。"""
    fs = cfg.fs
    if not fs.best_ckpt.exists():
        raise FileNotFoundError(
            f"找不到训练权重: {fs.best_ckpt}\n请先运行 `wakeup train`。"
        )
    tf = cfg.train
    model = WakeWordModel(tf.target_frames, tf.embedding_dim)
    model.load_state_dict(torch.load(fs.best_ckpt, map_location="cpu"))
    logger.info("从权重重新导出: %s", fs.best_ckpt)
    return export_onnx(cfg, model, simplify=simplify)


def _inline_external_data(onnx_path: Path) -> None:
    """把外部权重(.data)内联进单个 ONNX 文件，并清理旁文件。"""
    import onnx

    model = onnx.load(str(onnx_path))  # 默认会从同目录加载外部权重
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()
        logger.debug("已清理外部权重旁文件: %s", sidecar)


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
