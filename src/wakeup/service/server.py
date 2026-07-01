"""Controllable wake-word service over WebSocket."""

from __future__ import annotations

import asyncio
import json
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


class WakeWordService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._started_at = time.monotonic()
        self._clients: set[Any] = set()
        self._clients_lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_shutdown: asyncio.Event | None = None

        self._listen_flag = threading.Event()
        self._shutdown_flag = threading.Event()
        self._detector: WakeWordDetector | None = None
        self._worker: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: str | None = None
        self._worker_state = "starting"
        self._audio_restart_count = 0
        self._last_score = 0.0
        self._last_wake_ts: float | None = None

        if cfg.service.start_listening:
            self._listen_flag.set()

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
        self._schedule_broadcast(self.status())
        return None

    def stop_listening(self) -> None:
        if self._listen_flag.is_set():
            self._listen_flag.clear()
            logger.info("wake listening stopped")
        self._schedule_broadcast(self.status())

    def shutdown(self) -> None:
        logger.info("wake service shutdown requested")
        self._shutdown_flag.set()
        self._listen_flag.set()
        if self._loop is not None and self._async_shutdown is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._async_shutdown.set)

    async def handler(self, websocket) -> None:
        path = self._path(websocket)
        client = self._client_name(websocket)
        if path != self.cfg.service.ws_path:
            logger.warning("WebSocket rejected path=%s client=%s", path, client)
            await websocket.close(code=1008, reason="unsupported path")
            return

        if self._clients_lock is None:
            self._clients_lock = asyncio.Lock()
        async with self._clients_lock:
            self._clients.add(websocket)
        logger.info("WebSocket connected client=%s", client)
        try:
            await self._send_json(websocket, self.status())
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._send_json(websocket, {"type": p.TYPE_ERROR, "message": "binary input is not supported"})
                    continue
                try:
                    payload = json.loads(message)
                    if not isinstance(payload, dict):
                        raise ValueError("message must be a JSON object")
                    await self._dispatch(websocket, payload)
                except Exception as exc:
                    await self._send_json(websocket, {"type": p.TYPE_ERROR, "message": str(exc)})
        except ConnectionClosed:
            pass
        finally:
            if self._clients_lock is not None:
                async with self._clients_lock:
                    self._clients.discard(websocket)
            logger.info("WebSocket disconnected client=%s", client)

    async def _dispatch(self, websocket, payload: dict) -> None:
        cmd = str(payload.get("type") or payload.get("cmd") or "")
        if cmd == p.CMD_START:
            error = self.start_listening()
            await self._send_json(websocket, error or {"type": p.TYPE_ACK, "cmd": cmd, "ok": True})
        elif cmd == p.CMD_STOP:
            self.stop_listening()
            await self._send_json(websocket, {"type": p.TYPE_ACK, "cmd": cmd, "ok": True})
        elif cmd == p.CMD_STATUS:
            await self._send_json(websocket, self.status())
        elif cmd == p.CMD_PING:
            await self._send_json(websocket, {"type": p.TYPE_PONG})
        elif cmd == p.CMD_SHUTDOWN:
            await self._send_json(websocket, {"type": p.TYPE_ACK, "cmd": cmd, "ok": True})
            self.shutdown()
        else:
            await self._send_json(websocket, {"type": p.TYPE_ERROR, "message": f"unsupported command: {cmd}"})

    async def serve_forever_async(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._clients_lock = asyncio.Lock()
        self._async_shutdown = asyncio.Event()
        self._worker = threading.Thread(target=self._run_worker, name="detector", daemon=True)
        self._worker.start()

        async with websockets.serve(
            self.handler,
            self.cfg.service.host,
            self.cfg.service.port,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ):
            logger.info(
                "WakeUp WebSocket listening url=ws://%s:%d%s start_listening=%s",
                self.cfg.service.host,
                self.cfg.service.port,
                self.cfg.service.ws_path,
                self._listen_flag.is_set(),
            )
            await self._async_shutdown.wait()

        self._shutdown_flag.set()
        self._listen_flag.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        logger.info("WakeUp service stopped")

    def serve_forever(self) -> None:
        asyncio.run(self.serve_forever_async())

    def _run_worker(self) -> None:
        try:
            self._detector = WakeWordDetector(self.cfg)
            self._error = None
            self._worker_state = "idle"
            self._ready.set()
            self._schedule_broadcast(self.status())
        except Exception as exc:
            self._ready.clear()
            self._worker_state = "failed"
            self._error = f"wake word detector failed to load: {exc}"
            logger.error("wake word detector failed to load: %s", exc)
            self._schedule_broadcast({"type": p.TYPE_ERROR, "message": self._error})
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
                    self._schedule_broadcast(self.status())
                    while self._listen_flag.is_set() and not self._shutdown_flag.is_set():
                        frame = audio.read(timeout=0.5)
                        if frame is None:
                            continue
                        event = self._detector.process(frame)
                        self._last_score = float(getattr(self._detector, "last_active_score", self._last_score))
                        if event is not None:
                            self._last_wake_ts = event.ts
                            self._schedule_broadcast(
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
                self._schedule_broadcast({"type": p.TYPE_ERROR, "message": self._error})
                if self._shutdown_flag.wait(retry_seconds):
                    break
                retry_seconds = min(retry_seconds * 2.0, 10.0)

        self._worker_state = "stopped"
        logger.info("wake detector worker stopped")

    def _schedule_broadcast(self, message: dict) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)

    async def broadcast(self, message: dict) -> None:
        if self._clients_lock is None:
            return
        async with self._clients_lock:
            clients = list(self._clients)
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
            async with self._clients_lock:
                for client in dead:
                    self._clients.discard(client)

    async def _send_json(self, websocket, message: dict) -> None:
        await websocket.send(json.dumps(message, ensure_ascii=False))

    def _path(self, websocket) -> str | None:
        path = getattr(websocket, "path", None)
        if path is not None:
            return path
        request = getattr(websocket, "request", None)
        return getattr(request, "path", None)

    def _client_name(self, websocket) -> str:
        remote = websocket.remote_address
        if isinstance(remote, tuple) and len(remote) >= 2:
            return f"{remote[0]}:{remote[1]}"
        return str(remote)


def run_service(cfg: Config) -> None:
    WakeWordService(cfg).serve_forever()
