"""命令行入口。安装后通过 ``wakeup <子命令>`` 调用。

子命令：
    train      训练全流程（中文 TTS 正样本 -> 特征 -> 训练 -> 导出 ONNX）
    serve      启动常驻监听服务（可被其他程序通过 TCP 控制）
    ctl        控制运行中的服务：start / stop / status / shutdown
    events     连接服务并持续打印唤醒事件
    listen     前台直接监听（不起服务），用于现场调阈值
    devices    列出音频设备
    export-tf  把已有 ONNX 转成 TensorFlow SavedModel
"""

from __future__ import annotations

import argparse
import sys

from ._logging import get_logger, setup_logging
from .config import load_config

logger = get_logger(__name__)


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--config", "-c", default=None, help="YAML 配置文件路径")


def _add_client_opts(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--host", default=None, help="覆盖服务地址")
    sp.add_argument("--port", type=int, default=None, help="覆盖服务端口")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wakeup", description="本地中文语音唤醒（小元）")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    sp = sub.add_parser("train", help="训练并导出唤醒词模型")
    _add_common(sp)
    sp.add_argument("--skip-tts", action="store_true", help="跳过 TTS 合成（复用已有正样本）")
    sp.add_argument("--force-tts", action="store_true", help="强制重新合成正样本与特征")
    sp.add_argument("--export-tf", action="store_true", help="额外导出 TensorFlow 模型")
    sp.add_argument("--no-simplify", action="store_true", help="不简化 ONNX")

    # serve
    sp = sub.add_parser("serve", help="启动常驻监听服务")
    _add_common(sp)
    _add_client_opts(sp)
    sp.add_argument("--listen", action="store_true", help="启动后立即开始监听")

    # ctl
    sp = sub.add_parser("ctl", help="控制运行中的服务")
    _add_common(sp)
    _add_client_opts(sp)
    sp.add_argument("action", choices=["start", "stop", "status", "shutdown"])

    # events
    sp = sub.add_parser("events", help="持续打印唤醒事件")
    _add_common(sp)
    _add_client_opts(sp)

    # listen（前台直跑，调参用）
    sp = sub.add_parser("listen", help="前台直接监听（不起服务），调阈值用")
    _add_common(sp)
    sp.add_argument("--show-score", action="store_true", help="实时打印预测分")

    # devices
    sub.add_parser("devices", help="列出音频设备")

    # export-tf
    sp = sub.add_parser("export-tf", help="ONNX -> TensorFlow SavedModel")
    _add_common(sp)

    return parser


def _client_cfg(cfg, args):
    if getattr(args, "host", None):
        cfg.service.host = args.host
    if getattr(args, "port", None):
        cfg.service.port = args.port
    return cfg


# --------------------------------------------------------------------------- #
# 各子命令实现
# --------------------------------------------------------------------------- #
def cmd_train(args) -> int:
    from .training import run_training

    cfg = load_config(args.config)
    run_training(
        cfg,
        skip_tts=args.skip_tts,
        force_tts=args.force_tts,
        export_tf=args.export_tf,
        simplify=not args.no_simplify,
    )
    return 0


def cmd_serve(args) -> int:
    from .service.server import run_service

    cfg = load_config(args.config)
    cfg = _client_cfg(cfg, args)
    if args.listen:
        cfg.service.start_listening = True
    logger.info("启动服务，Ctrl+C 退出")
    try:
        run_service(cfg)
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，退出")
    return 0


def cmd_ctl(args) -> int:
    from .service.client import ServiceClient

    cfg = _client_cfg(load_config(args.config), args)
    try:
        with ServiceClient(cfg.service.host, cfg.service.port) as cli:
            resp = cli.command(args.action)
            print(resp)
    except OSError as exc:
        logger.error("无法连接服务 %s:%d —— %s", cfg.service.host, cfg.service.port, exc)
        return 1
    return 0


def cmd_events(args) -> int:
    from .service.client import ServiceClient

    cfg = _client_cfg(load_config(args.config), args)
    try:
        with ServiceClient(cfg.service.host, cfg.service.port) as cli:
            print(f"已连接 {cfg.service.host}:{cfg.service.port}，等待唤醒事件（Ctrl+C 退出）...")
            for msg in cli.messages():
                if msg.get("type") == "wake":
                    print(f"🔔 唤醒! model={msg['model']} score={msg['score']} ts={msg['ts']}")
                else:
                    print(f"· {msg}")
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        logger.error("无法连接服务: %s", exc)
        return 1
    return 0


def cmd_listen(args) -> int:
    from .service.audio import AudioInput
    from .service.detector import WakeWordDetector

    cfg = load_config(args.config)
    detector = WakeWordDetector(cfg)
    print("前台监听中（Ctrl+C 退出）。对着麦克风说「小元」试试。")
    try:
        with AudioInput(cfg) as audio:
            detector.reset()
            while True:
                frame = audio.read(timeout=0.5)
                if frame is None:
                    continue
                event = detector.process(frame)
                if args.show_score and detector.last_active_score > 0.01:
                    print(f"\r score={detector.last_active_score:.3f}", end="", flush=True)
                if event is not None:
                    print(f"\n🔔 唤醒! score={event.score:.3f}")
    except KeyboardInterrupt:
        print("\n退出")
    return 0


def cmd_devices(_args) -> int:
    from .service.audio import list_devices

    print(list_devices())
    return 0


def cmd_export_tf(args) -> int:
    from .training.export import export_tensorflow

    cfg = load_config(args.config)
    export_tensorflow(cfg)
    return 0


_DISPATCH = {
    "train": cmd_train,
    "serve": cmd_serve,
    "ctl": cmd_ctl,
    "events": cmd_events,
    "listen": cmd_listen,
    "devices": cmd_devices,
    "export-tf": cmd_export_tf,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    return _DISPATCH[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
