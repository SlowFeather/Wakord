"""轻量单元测试（不依赖音频/模型，可在 CI 直接跑）。

    pip install pytest && pytest
"""

import numpy as np
import pytest

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


def test_negative_windows_are_consecutive_not_repeated():
    """回归测试：负样本必须是连续帧窗口，而非单帧复制 16 次（曾导致误唤醒）。"""
    from wakeup.training.dataset import negative_windows

    # 构造每帧都不同的帧流：第 s 帧 = [s*96, s*96+1, ... ]
    stream = np.arange(200 * 96, dtype=np.float32).reshape(200, 96)
    rng = np.random.default_rng(0)
    out = negative_windows(stream, n_windows=10, target_frames=16, rng=rng)

    assert out.shape == (10, 16, 96)
    for window in out:
        # 窗口内各帧应互不相同（有时间结构），不是 16 个相同帧
        assert not np.allclose(window[0], window[1])
        # 且为流中的连续片段：相邻帧逐元素差恒为 96
        assert np.allclose(np.diff(window, axis=0), 96.0)


def test_protocol_roundtrip():
    from wakeup.service import protocol as p

    msg = {"type": "wake", "model": "小元", "score": 0.97}
    assert p.decode(p.encode(msg)) == msg


def test_split_indices_are_disjoint_and_stable():
    from wakeup.training.dataset import split_indices

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    train1, val1 = split_indices(20, 0.2, rng1)
    train2, val2 = split_indices(20, 0.2, rng2)

    assert set(train1).isdisjoint(set(val1))
    assert np.array_equal(train1, train2)
    assert np.array_equal(val1, val2)


def test_split_negative_stream_non_overlapping():
    from wakeup.training.dataset import split_negative_stream

    stream = np.arange(100 * 96, dtype=np.float32).reshape(100, 96)
    train, val = split_negative_stream(stream, 0.2, target_frames=16)

    assert len(train) + len(val) == len(stream)
    assert np.all(train[-1] != val[0])


def test_invalid_val_split_is_rejected():
    from wakeup.training.dataset import split_indices

    with pytest.raises(ValueError):
        split_indices(10, 0.0, np.random.default_rng(0))


def test_prepare_data_import_does_not_require_torch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from wakeup.training.pipeline import prepare_data

    assert callable(prepare_data)


def test_config_validation_rejects_bad_threshold(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("service:\n  threshold: 1.5\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(cfg_path)


def test_config_validation_rejects_bad_audio_queue_size(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("service:\n  audio_queue_size: 0\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(cfg_path)


def test_metrics_do_not_reward_all_negative_classifier():
    from wakeup.training.trainer import classification_metrics

    labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    scores = np.zeros_like(labels)
    metrics = classification_metrics(scores, labels, threshold=0.5)

    assert metrics["accuracy"] > 0.8
    assert metrics["f1"] == 0.0
    assert metrics["recall"] == 0.0


def test_download_sha256_mismatch_removes_file(tmp_path):
    from wakeup.data._download import verify_sha256

    path = tmp_path / "file.bin"
    path.write_bytes(b"bad")

    with pytest.raises(RuntimeError):
        verify_sha256(path, "0" * 64)
    assert not path.exists()


def test_safe_extract_rejects_path_traversal(tmp_path):
    import io
    import tarfile

    from wakeup.data.tts_generator import safe_extract_tar

    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        data = b"oops"
        info = tarfile.TarInfo("../evil.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    with tarfile.open(archive, "r") as tar:
        with pytest.raises(RuntimeError):
            safe_extract_tar(tar, tmp_path / "out")


def test_service_client_skips_greeting_and_broadcast():
    import socketserver
    import threading

    from wakeup.service import protocol as p
    from wakeup.service.client import ServiceClient

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.wfile.write(p.encode({"type": "status", "listening": False}))
            self.wfile.flush()
            raw = self.rfile.readline()
            cmd = p.decode(raw)["cmd"]
            self.wfile.write(p.encode({"type": "status", "listening": True}))
            self.wfile.write(p.encode({"type": "ack", "cmd": cmd, "ok": True}))
            self.wfile.flush()

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with ServiceClient(host, port, timeout=2.0) as client:
            assert client.initial_status["type"] == "status"
            assert client.command("start") == {"type": "ack", "cmd": "start", "ok": True}
    finally:
        server.shutdown()
        server.server_close()


def test_service_client_command_skips_mixed_push_messages():
    import socketserver
    import threading

    from wakeup.service import protocol as p
    from wakeup.service.client import ServiceClient

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.wfile.write(p.encode({"type": "status", "listening": False}))
            self.wfile.flush()
            raw = self.rfile.readline()
            cmd = p.decode(raw)["cmd"]
            self.wfile.write(p.encode({"type": "wake", "model": "xiaoyuan", "score": 0.9}))
            self.wfile.write(p.encode({"type": "status", "listening": True}))
            self.wfile.write(p.encode({"type": "ack", "cmd": cmd, "ok": True}))
            self.wfile.flush()

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with ServiceClient(host, port, timeout=2.0) as client:
            assert client.command("start") == {"type": "ack", "cmd": "start", "ok": True}
    finally:
        server.shutdown()
        server.server_close()


def test_service_start_returns_error_when_not_ready():
    from wakeup.service.server import WakeWordService

    service = WakeWordService(load_config())
    resp = service.start_listening()
    status = service.status()

    assert resp["type"] == "error"
    assert status["ready"] is False
    assert status["worker_alive"] is False
    assert "uptime_seconds" in status
    assert status["worker_state"] in {"starting", "failed"}
    assert status["audio_restart_count"] == 0


def test_eval_audio_dirs_with_fake_detector(tmp_path, monkeypatch):
    import soundfile as sf

    from wakeup.eval import evaluate_audio_dirs

    cfg = load_config()
    cfg.paths.base_dir = str(tmp_path / "artifacts")
    cfg.paths.model_path = str(tmp_path / "models" / "xiaoyuan.onnx")
    cfg.service.frame_samples = 1280
    cfg.service.threshold = 0.5
    pos = tmp_path / "positive"
    neg = tmp_path / "negative"
    pos.mkdir()
    neg.mkdir()
    sr = cfg.service.sample_rate
    sf.write(pos / "yes.wav", np.ones(sr, dtype=np.float32) * 0.1, sr)
    sf.write(neg / "no.wav", np.zeros(sr, dtype=np.float32), sr)

    class FakeDetector:
        def __init__(self, _cfg):
            self.last_active_score = 0.0

        def reset(self):
            self.last_active_score = 0.0

        def process(self, frame):
            self.last_active_score = 0.9 if np.abs(frame).mean() > 100 else 0.1
            return None

    monkeypatch.setattr("wakeup.eval.WakeWordDetector", FakeDetector)
    report = evaluate_audio_dirs(
        cfg,
        pos,
        neg,
        out_json=tmp_path / "report.json",
        out_csv=tmp_path / "report.csv",
    )

    assert report["counts"] == {"positive": 1, "negative": 1, "total": 2}
    assert report["metrics"]["f1"] == 1.0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.csv").exists()


def test_service_broadcasts_wake_with_fake_audio_and_detector(tmp_path, monkeypatch):
    import socket
    import threading
    import time

    from wakeup.service.detector import DetectionEvent
    from wakeup.service.server import WakeWordService
    from wakeup.service import protocol as p

    cfg = load_config()
    cfg.paths.base_dir = str(tmp_path / "artifacts")
    cfg.paths.model_path = str(tmp_path / "models" / "xiaoyuan.onnx")
    cfg.service.port = 0
    cfg.service.start_listening = False

    class FakeDetector:
        def __init__(self, _cfg):
            self.sent = False

        def reset(self):
            self.sent = False

        def process(self, _frame):
            if not self.sent:
                self.sent = True
                return DetectionEvent("xiaoyuan", 0.91, time.time())
            return None

    class FakeAudio:
        def __init__(self, _cfg):
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def read(self, timeout=0.5):
            self.n += 1
            time.sleep(0.01)
            return np.zeros(1280, dtype=np.int16)

    monkeypatch.setattr("wakeup.service.server.WakeWordDetector", FakeDetector)
    monkeypatch.setattr("wakeup.service.server.AudioInput", FakeAudio)

    service = WakeWordService(cfg)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 3.0
    while service._server is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert service._server is not None
    assert service._ready.wait(timeout=3.0)
    host, port = service._server.server_address

    with socket.create_connection((host, port), timeout=2.0) as sock:
        rfile = sock.makefile("rb")
        wfile = sock.makefile("wb")
        first = p.decode(rfile.readline())
        assert first["type"] == "status"
        wfile.write(p.encode({"cmd": "start"}))
        wfile.flush()
        msg = None
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            line = rfile.readline()
            if not line:
                break
            msg = p.decode(line)
            if msg.get("type") == "wake":
                break
        assert msg["type"] == "wake"
        assert msg["model"] == "xiaoyuan"

    service.shutdown()
    thread.join(timeout=3.0)


def test_service_retries_audio_after_start_failure(tmp_path, monkeypatch):
    import socket
    import threading
    import time

    from wakeup.service import protocol as p
    from wakeup.service.detector import DetectionEvent
    from wakeup.service.server import WakeWordService

    cfg = load_config()
    cfg.paths.base_dir = str(tmp_path / "artifacts")
    cfg.paths.model_path = str(tmp_path / "models" / "xiaoyuan.onnx")
    cfg.service.port = 0
    cfg.service.start_listening = True

    class FakeDetector:
        def __init__(self, _cfg):
            self.last_active_score = 0.0
            self.sent = False

        def reset(self):
            self.sent = False

        def process(self, _frame):
            self.last_active_score = 0.93
            if not self.sent:
                self.sent = True
                return DetectionEvent("xiaoyuan", 0.93, time.time())
            return None

    class FlakyAudio:
        starts = 0

        def __init__(self, _cfg):
            self.n = 0

        def __enter__(self):
            type(self).starts += 1
            if type(self).starts == 1:
                raise RuntimeError("device busy")
            return self

        def __exit__(self, *exc):
            return None

        def read(self, timeout=0.5):
            self.n += 1
            time.sleep(0.01)
            return np.zeros(1280, dtype=np.int16)

    monkeypatch.setattr("wakeup.service.server.WakeWordDetector", FakeDetector)
    monkeypatch.setattr("wakeup.service.server.AudioInput", FlakyAudio)

    service = WakeWordService(cfg)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 3.0
    while service._server is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert service._server is not None
    host, port = service._server.server_address

    try:
        with socket.create_connection((host, port), timeout=2.0) as sock:
            rfile = sock.makefile("rb")
            assert p.decode(rfile.readline())["type"] == "status"
            msg = None
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                line = rfile.readline()
                if not line:
                    break
                msg = p.decode(line)
                if msg.get("type") == "wake":
                    break
            assert msg["type"] == "wake"
            assert service.status()["audio_restart_count"] >= 1
            assert service.status()["worker_alive"] is True
    finally:
        service.shutdown()
        thread.join(timeout=3.0)


def test_service_shutdown_stops_worker_thread(tmp_path, monkeypatch):
    import threading
    import time

    from wakeup.service.server import WakeWordService

    cfg = load_config()
    cfg.paths.base_dir = str(tmp_path / "artifacts")
    cfg.paths.model_path = str(tmp_path / "models" / "xiaoyuan.onnx")
    cfg.service.port = 0
    cfg.service.start_listening = False

    class FakeDetector:
        def __init__(self, _cfg):
            self.last_active_score = 0.0

        def reset(self):
            return None

        def process(self, _frame):
            return None

    monkeypatch.setattr("wakeup.service.server.WakeWordDetector", FakeDetector)

    service = WakeWordService(cfg)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 3.0
    while service._server is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert service._server is not None
    assert service._ready.wait(timeout=3.0)
    service.shutdown()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert service._worker is not None
    assert not service._worker.is_alive()


def test_daemon_status_reports_pid_files(tmp_path):
    from wakeup.service.daemon import status_daemon

    cfg = load_config()
    cfg.paths.base_dir = str(tmp_path / "artifacts")
    status = status_daemon(cfg)

    assert status["process_alive"] is False
    assert status["service_responding"] is False
    assert status["pid_file"].endswith("wakeup.pid")
    assert status["meta_file"].endswith("wakeup-daemon.json")


def test_daemon_status_reports_stale_pid_file(tmp_path):
    from wakeup.service.daemon import daemon_paths, status_daemon

    cfg = load_config()
    cfg.paths.base_dir = str(tmp_path / "artifacts")
    paths = daemon_paths(cfg)
    paths.run_dir.mkdir(parents=True)
    paths.pid_file.write_text("999999", encoding="utf-8")

    status = status_daemon(cfg)

    assert status["pid_alive"] is False
    assert status["stale_pid_file"] is True


def test_cli_device_override_reaches_audio_config(monkeypatch):
    from wakeup import cli

    seen = {}

    class FakeDetector:
        def __init__(self, cfg):
            seen["detector_device"] = cfg.service.audio_device
            self.last_active_score = 0.0

        def reset(self):
            return None

        def process(self, _frame):
            raise KeyboardInterrupt

    class FakeAudio:
        def __init__(self, cfg):
            seen["audio_device"] = cfg.service.audio_device

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def read(self, timeout=0.5):
            raise KeyboardInterrupt

    monkeypatch.setattr("wakeup.service.detector.WakeWordDetector", FakeDetector)
    monkeypatch.setattr("wakeup.service.audio.AudioInput", FakeAudio)

    assert cli.main(["listen", "--device", "Mic 1"]) == 0
    assert seen == {"detector_device": "Mic 1", "audio_device": "Mic 1"}
