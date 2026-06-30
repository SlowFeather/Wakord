"""Controllable wake-word service with TCP and WebSocket frontends."""

from __future__ import annotations

import asyncio
import json
import socket
import socketserver
import threading
import time
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from .._logging import get_logger
from ..config import Config
from . import protocol as p
from .audio import AudioInput
from .detector import WakeWordDetector

logger = get_logger(__name__)


class _TcpClientConn:
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
        self._started_at = time.monotonic()
        self._tcp_clients: set[_TcpClientConn] = set()
        self._tcp_clients_lock = threading.Lock()
        self._ws_clients: set[Any] = set()
        self._ws_clients_lock: asyncio.Lock | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_shutdown: asyncio.Event | None = None

        self._listen_flag = threading.Event()
        self._shutdown_flag = threading.Event()
        self._detector: WakeWordDetector | None = None
        self._worker: threading.Thread | None = None
        self._server: socketserver.ThreadingTCPServer | None = None
        self._ws_thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: str | None = None
        self._worker_state = "starting"
        self._audio_restart_count = 0
        self._last_score = 0.0
        self._last_wake_ts: float | None = None

        if cfg.service.start_listening:
            self._listen_flag.set()

    def add_client(self, conn: _TcpClientConn) -> None:
        with self._tcp_clients_lock:
            self._tcp_clients.add(conn)

    def remove_client(self, conn: _TcpClientConn) -> None:
        with self._tcp_clients_lock:
            self._tcp_clients.discard(conn)

    def broadcast(self, message: dict) -> None:
        with self._tcp_clients_lock:
            dead = [client for client in self._tcp_clients if not client.send(message)]
            for client in dead:
                self._tcp_clients.discard(client)
        self._schedule_ws_broadcast(message)

    def status(self) -> dict:
        return {
            "type": p.TYPE_STATUS,
            "listening": self._listen_flag.is_set(),
            "model": self.cfg.service.model_name,
            "threshold": self.cfg.service.threshold,
            "ready": self._ready.is_set(),
            "worker_alive": self._worker.is_alive() if self._worker is not None else False,
            "error": self._error,
            "uptime_seconds": round(time.monotonic() - self._started_at, 3),
            "worker_state": self._worker_state,
            "last_error": self._error,
            "audio_restart_count": self._audio_restart_count,
            "last_score": round(self._last_score, 4),
            "last_wake_ts": self._last_wake_ts,
            "tcp_enabled": self.cfg.service.tcp_enabled,
            "ws_enabled": self.cfg.service.ws_enabled,
            "ws_path": self.cfg.service.ws_path,
        }

    def start_listening(self) -> dict | None:
        if not self._ready.is_set():
            message = self._error or "wake word detector is not ready"
            logger.warning("cannot start listening: %s", message)
            return {"type": p.TYPE_ERROR, "message": message}
        if not self._listen_flag.is_set():
            self._listen_flag.set()
            logger.info("wake listening started")
        self.broadcast(self.status())
        return None

    def stop_listening(self) -> None:
        if self._listen_flag.is_set():
            self._listen_flag.clear()
            logger.info("wake listening stopped")
        self.broadcast(self.status())

    def shutdown(self) -> None:
        logger.info("wake service shutdown requested")
        self._shutdown_flag.set()
        self._listen_flag.set()
        if self._server is not None:
            threading.Thread(target=self._server.shutdown, daemon=True).start()
        self._request_ws_shutdown()

    def _run_worker(self) -> None:
        try:
            self._detector = WakeWordDetector(self.cfg)
            self._error = None
            self._worker_state = "idle"
            self._ready.set()
            self.broadcast(self.status())
        except Exception as exc:
            self._ready.clear()
            self._worker_state = "failed"
            self._error = f"wake word detector failed to load: {exc}"
            logger.error("wake word detector failed to load: %s", exc)
            self.broadcast({"type": p.TYPE_ERROR, "message": self._error})
            return

        retry_seconds = 1.0
        while not self._shutdown_flag.is_set():
            self._worker_state = "idle"
            self._listen_flag.wait()
            if self._shutdown_flag.is_set():
                break

            try:
                self._worker_state = "opening_audio"
                with AudioInput(self.cfg) as audio:
                    retry_seconds = 1.0
                    self._error = None
                    self._detector.reset()
                    self._worker_state = "listening"
                    self.broadcast(self.status())
                    while self._listen_flag.is_set() and not self._shutdown_flag.is_set():
                        frame = audio.read(timeout=0.5)
                        if frame is None:
                            continue
                        event = self._detector.process(frame)
                        self._last_score = float(getattr(self._detector, "last_active_score", self._last_score))
                        if event is not None:
                            self._last_wake_ts = event.ts
                            self.broadcast(
                                {
                                    "type": p.TYPE_WAKE,
                                    "model": event.model,
                                    "score": round(event.score, 4),
                                    "ts": event.ts,
                                }
                            )
            except Exception as exc:
                self._audio_restart_count += 1
                self._error = f"audio/detection loop failed: {exc}"
                self._worker_state = "retrying_audio"
                logger.warning("audio/detection loop failed; retrying in %.1fs: %s", retry_seconds, exc)
                self.broadcast({"type": p.TYPE_ERROR, "message": self._error})
                if self._shutdown_flag.wait(retry_seconds):
                    break
                retry_seconds = min(retry_seconds * 2.0, 10.0)

        self._worker_state = "stopped"
        logger.info("wake detector worker stopped")

    async def ws_handler(self, websocket) -> None:
        path = self._ws_path(websocket)
        client = self._ws_client_name(websocket)
        if path != self.cfg.service.ws_path:
            logger.warning("WebSocket rejected path=%s client=%s", path, client)
            await websocket.close(code=1008, reason="unsupported path")
            return

        if self._ws_clients_lock is None:
            self._ws_clients_lock = asyncio.Lock()
        async with self._ws_clients_lock:
            self._ws_clients.add(websocket)
        logger.info("WebSocket connected client=%s", client)
        try:
            await self._ws_send_json(websocket, self.status())
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._ws_send_json(websocket, {"type": p.TYPE_ERROR, "message": "binary input is not supported"})
                    continue
                try:
                    payload = json.loads(message)
                    if not isinstance(payload, dict):
                        raise ValueError("message must be a JSON object")
                    await self._dispatch_ws(websocket, payload)
                except Exception as exc:
                    await self._ws_send_json(websocket, {"type": p.TYPE_ERROR, "message": str(exc)})
        except ConnectionClosed:
            pass
        finally:
            if self._ws_clients_lock is not None:
                async with self._ws_clients_lock:
                    self._ws_clients.discard(websocket)
            logger.info("WebSocket disconnected client=%s", client)

    async def _dispatch_ws(self, websocket, payload: dict) -> None:
        cmd = str(payload.get("type") or payload.get("cmd") or "")
        if cmd == p.CMD_START:
            error = self.start_listening()
            await self._ws_send_json(websocket, error or {"type": p.TYPE_ACK, "cmd": cmd, "ok": True})
        elif cmd == p.CMD_STOP:
            self.stop_listening()
            await self._ws_send_json(websocket, {"type": p.TYPE_ACK, "cmd": cmd, "ok": True})
        elif cmd == p.CMD_STATUS:
            await self._ws_send_json(websocket, self.status())
        elif cmd == p.CMD_PING:
            await self._ws_send_json(websocket, {"type": p.TYPE_PONG})
        elif cmd == p.CMD_SHUTDOWN:
            await self._ws_send_json(websocket, {"type": p.TYPE_ACK, "cmd": cmd, "ok": True})
            self.shutdown()
        else:
            await self._ws_send_json(websocket, {"type": p.TYPE_ERROR, "message": f"unsupported command: {cmd}"})

    async def ws_serve_forever(self) -> None:
        self._ws_loop = asyncio.get_running_loop()
        self._ws_clients_lock = asyncio.Lock()
        self._ws_shutdown = asyncio.Event()
        async with websockets.serve(
            self.ws_handler,
            self.cfg.service.host,
            self.cfg.service.ws_port,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ):
            logger.info(
                "WakeUp WebSocket listening url=ws://%s:%d%s",
                self.cfg.service.host,
                self.cfg.service.ws_port,
                self.cfg.service.ws_path,
            )
            await self._ws_shutdown.wait()

    def _run_ws_server(self) -> None:
        try:
            asyncio.run(self.ws_serve_forever())
        except Exception as exc:
            if not self._shutdown_flag.is_set():
                logger.exception("WakeUp WebSocket service failed: %s", exc)

    async def _ws_broadcast(self, message: dict) -> None:
        if self._ws_clients_lock is None:
            return
        async with self._ws_clients_lock:
            clients = list(self._ws_clients)
        if not clients:
            return

        payload = json.dumps(message, ensure_ascii=False)
        dead = []
        for client in clients:
            try:
                await client.send(payload)
            except Exception:
                dead.append(client)
        if dead:
            async with self._ws_clients_lock:
                for client in dead:
                    self._ws_clients.discard(client)

    def _schedule_ws_broadcast(self, message: dict) -> None:
        if self._ws_loop is None or self._ws_loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._ws_broadcast(message), self._ws_loop)

    async def _ws_send_json(self, websocket, message: dict) -> None:
        await websocket.send(json.dumps(message, ensure_ascii=False))

    def _ws_path(self, websocket) -> str | None:
        path = getattr(websocket, "path", None)
        if path is not None:
            return path
        request = getattr(websocket, "request", None)
        return getattr(request, "path", None)

    def _ws_client_name(self, websocket) -> str:
        remote = websocket.remote_address
        if isinstance(remote, tuple) and len(remote) >= 2:
            return f"{remote[0]}:{remote[1]}"
        return str(remote)

    def serve_forever(self) -> None:
        self._worker = threading.Thread(target=self._run_worker, name="detector", daemon=True)
        self._worker.start()

        if self.cfg.service.ws_enabled:
            self._ws_thread = threading.Thread(target=self._run_ws_server, name="wakeup-ws", daemon=True)
            self._ws_thread.start()

        if self.cfg.service.tcp_enabled:
            self._serve_tcp_forever()
        else:
            try:
                while not self._shutdown_flag.wait(0.25):
                    pass
            finally:
                self._cleanup()

    def _serve_tcp_forever(self) -> None:
        service = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                conn = _TcpClientConn(self.wfile)
                service.add_client(conn)
                conn.send(service.status())
                peer = self.client_address
                logger.info("TCP client connected %s", peer)
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
                    logger.info("TCP client disconnected %s", peer)

            def _dispatch(self, conn: _TcpClientConn, raw: bytes):
                try:
                    msg = p.decode(raw)
                    cmd = msg.get("cmd")
                except Exception:
                    conn.send({"type": p.TYPE_ERROR, "message": "invalid JSON"})
                    return

                if cmd == p.CMD_START:
                    error = service.start_listening()
                    conn.send(error or {"type": p.TYPE_ACK, "cmd": cmd, "ok": True})
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
                    conn.send({"type": p.TYPE_ERROR, "message": f"unknown command: {cmd}"})

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True
            address_family = socket.AF_INET

        host, port = self.cfg.service.host, self.cfg.service.port
        self._server = Server((host, port), Handler)
        logger.info("WakeUp TCP listening %s:%d start_listening=%s", host, port, self._listen_flag.is_set())
        try:
            self._server.serve_forever()
        finally:
            self._server.server_close()
            self._cleanup()

    def _cleanup(self) -> None:
        self._shutdown_flag.set()
        self._listen_flag.set()
        self._request_ws_shutdown()
        with self._tcp_clients_lock:
            self._tcp_clients.clear()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        if self._ws_thread is not None and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=3.0)
        logger.info("WakeUp service stopped")

    def _request_ws_shutdown(self) -> None:
        if self._ws_loop is None or self._ws_shutdown is None or self._ws_loop.is_closed():
            return
        try:
            self._ws_loop.call_soon_threadsafe(self._ws_shutdown.set)
        except RuntimeError:
            return


def run_service(cfg: Config) -> None:
    WakeWordService(cfg).serve_forever()
