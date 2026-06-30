from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import websockets

from . import protocol as p


class WsServiceClient:
    def __init__(self, url: str = "ws://127.0.0.1:8766/v1/wake/ws", timeout: float = 5.0):
        self.url = url
        self.timeout = timeout
        self._ws = None
        self.initial_status: dict | None = None

    async def connect(self) -> "WsServiceClient":
        self._ws = await websockets.connect(self.url, open_timeout=self.timeout, max_size=None)
        raw = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout)
        self.initial_status = _decode(raw)
        return self

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
        self._ws = None

    async def __aenter__(self) -> "WsServiceClient":
        return await self.connect()

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def send_json(self, message: dict) -> None:
        if self._ws is None:
            raise ConnectionError("service client is not connected")
        await self._ws.send(json.dumps(message, ensure_ascii=False))

    async def recv(self) -> dict | None:
        if self._ws is None:
            raise ConnectionError("service client is not connected")
        raw = await self._ws.recv()
        return _decode(raw)

    async def messages(self) -> AsyncIterator[dict]:
        if self._ws is None:
            raise ConnectionError("service client is not connected")
        async for raw in self._ws:
            if isinstance(raw, bytes):
                continue
            yield json.loads(raw)

    async def command(self, cmd: str) -> dict | None:
        await self.send_json({"type": cmd})
        deadline = asyncio.get_running_loop().time() + self.timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for response to {cmd!r}")
            msg = await asyncio.wait_for(self.recv(), timeout=remaining)
            if msg is None:
                raise ConnectionError("service closed the connection")
            typ = msg.get("type")
            if typ == p.TYPE_ERROR:
                return msg
            if cmd == p.CMD_STATUS and typ == p.TYPE_STATUS:
                return msg
            if typ == p.TYPE_ACK and msg.get("cmd") == cmd:
                return msg
            if cmd == p.CMD_PING and typ == p.TYPE_PONG:
                return msg

    async def start(self) -> dict | None:
        return await self.command(p.CMD_START)

    async def stop(self) -> dict | None:
        return await self.command(p.CMD_STOP)

    async def status(self) -> dict | None:
        return await self.command(p.CMD_STATUS)

    async def shutdown(self) -> dict | None:
        return await self.command(p.CMD_SHUTDOWN)


def wake_ws_url(host: str, port: int, path: str) -> str:
    return f"ws://{host}:{port}{path}"


def _decode(raw) -> dict:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)
