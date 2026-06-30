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
import asyncio
import sys

from ._logging import get_logger, setup_logging
from .config import load_config

logger = get_logger(__name__)


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--config", "-c", default=None, help="YAML 配置文件路径")


def _add_client_opts(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--host", default=None, help="覆盖服务地址")
    sp.add_argument("--port", type=int, default=None, help="覆盖服务端口")
    sp.add_argument("--transport", choices=["ws", "tcp"], default="ws", help="控制协议，默认 WebSocket")
    sp.add_argument("--ws-port", type=int, default=None, help="覆盖 WebSocket 端口")
    sp.add_argument("--ws-path", default=None, help="覆盖 WebSocket 路径")


def _add_audio_device_opt(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--device", default=None, help="覆盖音频输入设备 ID 或名称")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wakeup", description="本地中文语音唤醒（小元）")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    # train（一条龙 = prepare + fit）
    sp = sub.add_parser("train", help="一条龙训练并导出（= prepare + fit）")
    _add_common(sp)
    sp.add_argument("--skip-tts", action="store_true", help="跳过 TTS 合成（复用已有正样本）")
    sp.add_argument("--force-tts", action="store_true", help="强制重新合成正样本与特征")
    sp.add_argument("--gen-voices", action="store_true",
                    help="训练前用 Edge TTS 多音色扩充正样本（需联网，无需录音）")
    sp.add_argument("--voices-count", type=int, default=None,
                    help="--gen-voices 时的合成条数上限（默认全部组合 ~165）")
    sp.add_argument("--force-features", action="store_true",
                    help="强制重新提取特征（新增录音/音色后用）")
    sp.add_argument("--epochs", type=int, default=None, help="临时覆盖训练轮数（快速试跑用）")
    sp.add_argument("--export-tf", action="store_true", help="额外导出 TensorFlow 模型")
    sp.add_argument("--no-simplify", action="store_true", help="不简化 ONNX")
    sp.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None,
                    help="训练设备；默认 auto（有 GPU 自动用 GPU）")

    # prepare（只做数据+特征，慢；样本变化时才重跑）
    sp = sub.add_parser("prepare", help="只做数据准备+特征提取并缓存（慢，样本变化时才重跑）")
    _add_common(sp)
    sp.add_argument("--skip-tts", action="store_true", help="跳过 TTS 合成（复用已有正样本）")
    sp.add_argument("--force-tts", action="store_true", help="强制重新合成正样本与特征")
    sp.add_argument("--gen-voices", action="store_true",
                    help="用 Edge TTS 多音色扩充正样本（需联网，无需录音）")
    sp.add_argument("--voices-count", type=int, default=None,
                    help="--gen-voices 时的合成条数上限（默认全部组合 ~165）")
    sp.add_argument("--force-features", action="store_true",
                    help="强制重新提取特征（新增录音/音色后用）")

    # fit（只做训练+导出，快；调参反复跑这个）
    sp = sub.add_parser("fit", help="从已缓存的特征训练+导出（快，调参反复跑）")
    _add_common(sp)
    sp.add_argument("--epochs", type=int, default=None, help="临时覆盖训练轮数（快速试跑用）")
    sp.add_argument("--export-tf", action="store_true", help="额外导出 TensorFlow 模型")
    sp.add_argument("--no-simplify", action="store_true", help="不简化 ONNX")
    sp.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None,
                    help="训练设备；默认 auto（有 GPU 自动用 GPU）")

    # serve
    sp = sub.add_parser("serve", help="启动常驻监听服务")
    _add_common(sp)
    _add_client_opts(sp)
    _add_audio_device_opt(sp)
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
    _add_audio_device_opt(sp)
    sp.add_argument("--show-score", action="store_true", help="实时打印预测分")
    sp.add_argument("--debug", action="store_true",
                    help="诊断模式：不门控、每帧都跑模型，连续打印 VAD/原始分/麦克风电平")
    sp.add_argument("--threshold", type=float, default=None, help="临时覆盖触发阈值")
    sp.add_argument("--vad", choices=["auto", "silero", "webrtc", "energy", "none"],
                    default=None, help="临时覆盖 VAD 后端")

    # record（录制真实唤醒词样本，few-shot 个性化）
    sp = sub.add_parser("record", help="用麦克风录制真实「小元」样本，混入训练提升召回")
    _add_common(sp)
    sp.add_argument("--count", type=int, default=30, help="录制条数（默认 30）")
    sp.add_argument("--seconds", type=float, default=1.5, help="每条时长秒（默认 1.5）")

    # eval（真实音频验收）
    sp = sub.add_parser("eval-record", help="录制真实验收音频到 positive/negative 目录")
    _add_common(sp)
    sp.add_argument("label", choices=["positive", "negative"], help="positive=唤醒词，negative=非唤醒/环境声")
    sp.add_argument("--count", type=int, default=20, help="录制条数（默认 20）")
    sp.add_argument("--seconds", type=float, default=3.0, help="每条时长秒（默认 3.0）")
    sp.add_argument("--out-dir", default=None, help="覆盖输出目录")

    sp = sub.add_parser("eval", help="用真实正/负音频目录离线验收模型")
    _add_common(sp)
    sp.add_argument("--positive-dir", default=None, help="正样本目录（默认 artifacts/data/eval/positive）")
    sp.add_argument("--negative-dir", default=None, help="负样本目录（默认 artifacts/data/eval/negative）")
    sp.add_argument("--threshold", type=float, default=None, help="临时覆盖验收阈值")
    sp.add_argument("--out-json", default=None, help="输出 JSON 报告路径")
    sp.add_argument("--out-csv", default=None, help="输出逐条 CSV 路径")

    # gen-voices（Edge TTS 多音色合成，扩充 TTS 多样性）
    sp = sub.add_parser("gen-voices", help="用 Edge TTS 多音色合成「小元」，扩充正样本多样性")
    _add_common(sp)
    sp.add_argument("--count", type=int, default=None, help="生成条数上限（默认全部组合 ~165）")

    # daemon（后台进程 / 自启动）
    sp = sub.add_parser("daemon", help="后台守护进程管理")
    _add_common(sp)
    _add_client_opts(sp)
    sp.add_argument("action", choices=["start", "stop", "status", "install", "uninstall"])
    sp.add_argument("--listen", action="store_true", help="start/install 后立即开始监听")
    sp.add_argument("--wait", type=float, default=10.0, help="start 等待服务就绪的秒数")

    # devices
    sp = sub.add_parser("devices", help="列出音频设备")
    _add_common(sp)
    _add_audio_device_opt(sp)

    # export（从已训练权重重新导出 ONNX，不重训）
    sp = sub.add_parser("export", help="从已训练权重(best.pth)重新导出 ONNX（不重训）")
    _add_common(sp)
    sp.add_argument("--no-simplify", action="store_true", help="不简化 ONNX")

    # export-tf
    sp = sub.add_parser("export-tf", help="ONNX -> TensorFlow SavedModel")
    _add_common(sp)

    return parser


def _client_cfg(cfg, args):
    if getattr(args, "host", None):
        cfg.service.host = args.host
    if getattr(args, "port", None):
        if getattr(args, "transport", "tcp") == "ws":
            cfg.service.ws_port = args.port
        else:
            cfg.service.port = args.port
    if getattr(args, "ws_port", None):
        cfg.service.ws_port = args.ws_port
    if getattr(args, "ws_path", None):
        cfg.service.ws_path = args.ws_path
    return cfg


def _audio_cfg(cfg, args):
    if getattr(args, "device", None) is not None:
        cfg.service.audio_device = args.device
    return cfg


# --------------------------------------------------------------------------- #
# 各子命令实现
# --------------------------------------------------------------------------- #
def cmd_train(args) -> int:
    from .training import run_training

    cfg = load_config(args.config)
    if args.device:
        cfg.train.device = args.device
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    run_training(
        cfg,
        skip_tts=args.skip_tts,
        force_tts=args.force_tts,
        gen_voices=args.gen_voices,
        voices_count=args.voices_count,
        force_features=args.force_features,
        export_tf=args.export_tf,
        simplify=not args.no_simplify,
    )
    return 0


def cmd_prepare(args) -> int:
    from .training import prepare_data

    cfg = load_config(args.config)
    prepare_data(
        cfg,
        skip_tts=args.skip_tts,
        force_tts=args.force_tts,
        gen_voices=args.gen_voices,
        voices_count=args.voices_count,
        force_features=args.force_features,
    )
    return 0


def cmd_fit(args) -> int:
    from .training import fit

    cfg = load_config(args.config)
    if args.device:
        cfg.train.device = args.device
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    fit(cfg, export_tf=args.export_tf, simplify=not args.no_simplify)
    return 0


def cmd_serve(args) -> int:
    from .service.server import run_service

    cfg = load_config(args.config)
    cfg = _client_cfg(cfg, args)
    cfg = _audio_cfg(cfg, args)
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
    from .service.ws_client import wake_ws_url

    cfg = _client_cfg(load_config(args.config), args)
    if args.transport == "ws":
        url = wake_ws_url(cfg.service.host, cfg.service.ws_port, cfg.service.ws_path)
        try:
            resp = asyncio.run(_ws_command(url, args.action))
            print(resp)
        except OSError as exc:
            logger.error("无法连接 WebSocket 服务 %s —— %s", url, exc)
            return 1
        return 0
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
    from .service.ws_client import wake_ws_url

    cfg = _client_cfg(load_config(args.config), args)
    if args.transport == "ws":
        url = wake_ws_url(cfg.service.host, cfg.service.ws_port, cfg.service.ws_path)
        try:
            asyncio.run(_print_ws_events(url))
        except KeyboardInterrupt:
            pass
        except OSError as exc:
            logger.error("无法连接 WebSocket 服务 %s —— %s", url, exc)
            return 1
        return 0
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


async def _ws_command(url: str, action: str) -> dict | None:
    from .service.ws_client import WsServiceClient

    async with WsServiceClient(url) as cli:
        return await cli.command(action)


async def _print_ws_events(url: str) -> None:
    from .service.ws_client import WsServiceClient

    async with WsServiceClient(url) as cli:
        print(f"已连接 {url}，等待唤醒事件（Ctrl+C 退出）...")
        async for msg in cli.messages():
            if msg.get("type") == "wake":
                print(f"🔔 唤醒! model={msg['model']} score={msg['score']} ts={msg['ts']}")
            else:
                print(f"· {msg}")


def cmd_listen(args) -> int:
    from .service.audio import AudioInput
    from .service.detector import WakeWordDetector

    cfg = load_config(args.config)
    cfg = _audio_cfg(cfg, args)
    if args.threshold is not None:
        cfg.service.threshold = args.threshold
    if args.vad is not None:
        cfg.service.vad_backend = args.vad
    detector = WakeWordDetector(cfg)
    if args.debug:
        return _listen_debug(cfg, detector, AudioInput)
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


def _listen_debug(cfg, detector, AudioInput) -> int:
    """诊断模式：不做 VAD 门控，每帧都跑唤醒模型（缓冲始终预热），连续打印
    VAD 判定 / 原始分 / 运行最高分 / 麦克风电平(RMS)。用于区分"VAD 没抓到人声"
    与"模型不认这把嗓音"。"""
    import numpy as np

    thr = cfg.service.threshold
    print(f"诊断监听中（Ctrl+C 退出）。阈值={thr:.2f}，VAD={detector.vad.kind}。")
    print("对麦克风说「小元」，观察 score 峰值；安静时看 score 是否仍乱跳。\n")
    try:
        with AudioInput(cfg) as audio:
            detector.reset()
            if hasattr(detector.oww, "reset"):
                detector.oww.reset()
            peak = 0.0
            while True:
                frame = audio.read(timeout=0.5)
                if frame is None:
                    continue
                speech = detector.vad.is_speech(frame)
                raw = detector._predict(frame)  # 每帧都跑，缓冲保持预热
                peak = max(peak, raw)
                rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))
                mark = " 🔔" if raw >= thr else ""
                print(f"\r vad={'V' if speech else '.'} score={raw:6.3f} "
                      f"peak={peak:.3f} mic_rms={rms:.3f}{mark}   ", end="", flush=True)
    except KeyboardInterrupt:
        print(f"\n退出。本次最高分 peak={peak:.3f}")
    return 0


def cmd_record(args) -> int:
    from .data.recorder import record_samples

    cfg = load_config(args.config)
    record_samples(cfg, count=args.count, seconds=args.seconds)
    return 0


def cmd_eval_record(args) -> int:
    from pathlib import Path

    from .data.recorder import record_samples

    cfg = load_config(args.config)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.label == "positive":
        out_dir = cfg.fs.eval_positive_dir
    else:
        out_dir = cfg.fs.eval_negative_dir
    record_samples(cfg, count=args.count, seconds=args.seconds, out_dir=out_dir)
    return 0


def cmd_eval(args) -> int:
    from pathlib import Path

    from .eval import evaluate_audio_dirs

    cfg = load_config(args.config)
    pos = Path(args.positive_dir) if args.positive_dir else cfg.fs.eval_positive_dir
    neg = Path(args.negative_dir) if args.negative_dir else cfg.fs.eval_negative_dir
    out_json = Path(args.out_json) if args.out_json else cfg.fs.model_dir / "real_eval.json"
    out_csv = Path(args.out_csv) if args.out_csv else cfg.fs.model_dir / "real_eval.csv"
    report = evaluate_audio_dirs(
        cfg,
        pos,
        neg,
        threshold=args.threshold,
        out_json=out_json,
        out_csv=out_csv,
    )
    m = report["metrics"]
    print(
        "真实音频验收完成："
        f"total={report['counts']['total']} "
        f"precision={m['precision']:.3f} recall={m['recall']:.3f} "
        f"f1={m['f1']:.3f} fpr={m['false_positive_rate']:.3f} "
        f"threshold={m['threshold']:.2f}"
    )
    print(f"JSON: {out_json}")
    print(f"CSV: {out_csv}")
    return 0


def cmd_gen_voices(args) -> int:
    from .data.tts_edge import generate_edge_samples

    cfg = load_config(args.config)
    generate_edge_samples(cfg, count=args.count)
    return 0


def cmd_daemon(args) -> int:
    from .service.daemon import (
        as_json,
        install_autostart,
        start_daemon,
        status_daemon,
        stop_daemon,
        uninstall_autostart,
    )

    cfg = _client_cfg(load_config(args.config), args)
    if args.action == "start":
        result = start_daemon(cfg, config_path=args.config, listen=args.listen, wait_seconds=args.wait)
    elif args.action == "stop":
        result = stop_daemon(cfg)
    elif args.action == "status":
        result = status_daemon(cfg)
    elif args.action == "install":
        result = install_autostart(cfg, config_path=args.config, listen=args.listen)
    else:
        result = uninstall_autostart()
    print(as_json(result))
    return 0


def cmd_devices(args) -> int:
    from .service.audio import list_devices

    cfg = _audio_cfg(load_config(args.config), args)
    if cfg.service.audio_device is not None:
        print(f"当前覆盖输入设备: {cfg.service.audio_device}")
    print(list_devices())
    return 0


def cmd_export(args) -> int:
    from .training.export import export_from_checkpoint

    cfg = load_config(args.config)
    export_from_checkpoint(cfg, simplify=not args.no_simplify)
    return 0


def cmd_export_tf(args) -> int:
    from .training.export import export_tensorflow

    cfg = load_config(args.config)
    export_tensorflow(cfg)
    return 0


_DISPATCH = {
    "train": cmd_train,
    "prepare": cmd_prepare,
    "fit": cmd_fit,
    "serve": cmd_serve,
    "ctl": cmd_ctl,
    "events": cmd_events,
    "listen": cmd_listen,
    "record": cmd_record,
    "eval-record": cmd_eval_record,
    "eval": cmd_eval,
    "gen-voices": cmd_gen_voices,
    "daemon": cmd_daemon,
    "devices": cmd_devices,
    "export": cmd_export,
    "export-tf": cmd_export_tf,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    return _DISPATCH[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
