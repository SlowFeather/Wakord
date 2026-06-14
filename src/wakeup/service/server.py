"""可被外部程序控制的常驻唤醒词服务。

架构：
* 一个 TCP 控制服务（ThreadingTCPServer），按 JSON-lines 协议收命令、回事件。
* 一个后台检测线程：仅在「监听中」时打开麦克风并跑检测；
  「停止监听」时**关闭麦克风**并阻塞等待，做到真正省电、随时可控。
* 唤醒命中后向所有已连接客户端广播 ``wake`` 事件。
"""

from __future__ import annotations

import socket
import socketserver
import threading
import time

from .._logging import get_logger
from ..config import Config
from . import protocol as p
from .audio import AudioInput
from .detector import WakeWordDetector

logger = get_logger(__name__)


class _ClientConn:
    """封装一个客户端连接的写端，带锁避免多线程交错写。"""

    def __init__(self, wfile):
        self._wfile = wfile
        self._lock = threading.Lock()

    def send(self, message: dict) -> bool:
        try:
            with self._lock:
                self._wfile.write(p.encode(message))
                self._wfile.flush()
            return True
        except (BrokenPipeError, ConnectionError, OSError, ValueError):
            return False


class WakeWordService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._clients: set[_ClientConn] = set()
        self._clients_lock = threading.Lock()

        self._listen_flag = threading.Event()       # 置位=监听中
        self._shutdown_flag = threading.Event()     # 置位=进程退出
        self._detector: WakeWordDetector | None = None
        self._worker: threading.Thread | None = None
        self._server: socketserver.ThreadingTCPServer | None = None

        if cfg.service.start_listening:
            self._listen_flag.set()

    # ---------------- 客户端管理 / 广播 ----------------
    def add_client(self, conn: _ClientConn) -> None:
        with self._clients_lock:
            self._clients.add(conn)

    def remove_client(self, conn: _ClientConn) -> None:
        with self._clients_lock:
            self._clients.discard(conn)

    def broadcast(self, message: dict) -> None:
        with self._clients_lock:
            dead = [c for c in self._clients if not c.send(message)]
            for c in dead:
                self._clients.discard(c)

    def status(self) -> dict:
        return {
            "type": p.TYPE_STATUS,
            "listening": self._listen_flag.is_set(),
            "model": self.cfg.service.model_name,
            "threshold": self.cfg.service.threshold,
        }

    # ---------------- 控制命令 ----------------
    def start_listening(self) -> None:
        if not self._listen_flag.is_set():
            self._listen_flag.set()
            logger.info("开始监听")
        self.broadcast(self.status())

    def stop_listening(self) -> None:
        if self._listen_flag.is_set():
            self._listen_flag.clear()
            logger.info("停止监听")
        self.broadcast(self.status())

    def shutdown(self) -> None:
        logger.info("收到关闭指令")
        self._shutdown_flag.set()
        self._listen_flag.set()  # 唤醒可能在阻塞等待的 worker
        if self._server is not None:
            threading.Thread(target=self._server.shutdown, daemon=True).start()

    # ---------------- 后台检测线程 ----------------
    def _run_worker(self) -> None:
        # 模型加载较慢，放到线程里做，避免阻塞服务启动
        try:
            self._detector = WakeWordDetector(self.cfg)
        except Exception as exc:
            logger.error("唤醒词模型加载失败: %s", exc)
            self.broadcast({"type": p.TYPE_ERROR, "message": f"模型加载失败: {exc}"})
            return

        while not self._shutdown_flag.is_set():
            # 未监听时阻塞等待 —— 不占麦克风、不耗算力
            self._listen_flag.wait()
            if self._shutdown_flag.is_set():
                break

            try:
                with AudioInput(self.cfg) as audio:
                    self._detector.reset()
                    while self._listen_flag.is_set() and not self._shutdown_flag.is_set():
                        frame = audio.read(timeout=0.5)
                        if frame is None:
                            continue
                        event = self._detector.process(frame)
                        if event is not None:
                            self.broadcast({
                                "type": p.TYPE_WAKE,
                                "model": event.model,
                                "score": round(event.score, 4),
                                "ts": event.ts,
                            })
            except Exception as exc:
                logger.error("检测循环异常: %s", exc)
                self.broadcast({"type": p.TYPE_ERROR, "message": str(exc)})
                # 出错后避免疯狂重试
                time.sleep(1.0)

        logger.info("检测线程退出")

    # ---------------- 启动 / 阻塞运行 ----------------
    def serve_forever(self) -> None:
        self._worker = threading.Thread(target=self._run_worker, name="detector",
                                        daemon=True)
        self._worker.start()

        service = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                conn = _ClientConn(self.wfile)
                service.add_client(conn)
                conn.send(service.status())
                peer = self.client_address
                logger.info("客户端接入: %s", peer)
                try:
                    for raw in self.rfile:
                        raw = raw.strip()
                        if not raw:
                            continue
                        self._dispatch(conn, raw)
                except (ConnectionError, OSError):
                    pass
                finally:
                    service.remove_client(conn)
                    logger.info("客户端断开: %s", peer)

            def _dispatch(self, conn: _ClientConn, raw: bytes):
                try:
                    msg = p.decode(raw)
                    cmd = msg.get("cmd")
                except Exception:
                    conn.send({"type": p.TYPE_ERROR, "message": "非法 JSON"})
                    return

                if cmd == p.CMD_START:
                    service.start_listening()
                    conn.send({"type": p.TYPE_ACK, "cmd": cmd, "ok": True})
                elif cmd == p.CMD_STOP:
                    service.stop_listening()
                    conn.send({"type": p.TYPE_ACK, "cmd": cmd, "ok": True})
                elif cmd == p.CMD_STATUS:
                    conn.send(service.status())
                elif cmd == p.CMD_PING:
                    conn.send({"type": p.TYPE_PONG})
                elif cmd == p.CMD_SHUTDOWN:
                    conn.send({"type": p.TYPE_ACK, "cmd": cmd, "ok": True})
                    service.shutdown()
                else:
                    conn.send({"type": p.TYPE_ERROR, "message": f"未知命令: {cmd}"})

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True
            address_family = socket.AF_INET

        host, port = self.cfg.service.host, self.cfg.service.port
        self._server = Server((host, port), Handler)
        logger.info("控制接口监听 %s:%d（start_listening=%s）",
                    host, port, self._listen_flag.is_set())
        try:
            self._server.serve_forever()
        finally:
            self._server.server_close()
            self._shutdown_flag.set()
            self._listen_flag.set()
            logger.info("服务已停止")


def run_service(cfg: Config) -> None:
    WakeWordService(cfg).serve_forever()
