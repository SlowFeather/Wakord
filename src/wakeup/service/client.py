"""控制客户端：连接服务、发命令、接收事件。

既可作为库在你自己的 Python 程序里 import 使用，也可经 CLI 调用：
    wakeup ctl start / stop / status / shutdown
    wakeup events            # 持续打印唤醒事件
"""

from __future__ import annotations

import socket
import time
from typing import Callable, Iterator

from .._logging import get_logger
from . import protocol as p

logger = get_logger(__name__)


class ServiceClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._rfile = None
        self._wfile = None
        self.initial_status: dict | None = None

    def connect(self) -> "ServiceClient":
        self._sock = socket.create_connection((self.host, self.port), self.timeout)
        self._sock.settimeout(self.timeout)
        self._rfile = self._sock.makefile("rb")
        self._wfile = self._sock.makefile("wb")
        self.initial_status = self.recv()
        self._sock.settimeout(None)
        return self

    def close(self) -> None:
        for f in (self._rfile, self._wfile):
            try:
                if f:
                    f.close()
            except Exception:
                pass
        if self._sock:
            self._sock.close()
        self._sock = None

    def __enter__(self) -> "ServiceClient":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- 收发 ----
    def send(self, message: dict) -> None:
        if self._wfile is None:
            raise ConnectionError("service client is not connected")
        self._wfile.write(p.encode(message))
        self._wfile.flush()

    def recv(self) -> dict | None:
        if self._rfile is None:
            raise ConnectionError("service client is not connected")
        line = self._rfile.readline()
        if not line:
            return None
        return p.decode(line)

    def messages(self) -> Iterator[dict]:
        """持续迭代服务端推送的消息，直到连接断开。"""
        while True:
            msg = self.recv()
            if msg is None:
                break
            yield msg

    # ---- 便捷命令（发完读一条响应）----
    def command(self, cmd: str) -> dict | None:
        if self._sock is None:
            raise ConnectionError("service client is not connected")
        self.send({"cmd": cmd})
        deadline = time.monotonic() + self.timeout
        old_timeout = self._sock.gettimeout()
        self._sock.settimeout(self.timeout)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for response to {cmd!r}")
                self._sock.settimeout(remaining)
                msg = self.recv()
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
                logger.debug("忽略命令响应前的推送消息: %s", msg)
        except socket.timeout as exc:
            raise TimeoutError(f"timed out waiting for response to {cmd!r}") from exc
        finally:
            self._sock.settimeout(old_timeout)

    def start(self) -> dict | None:
        return self.command(p.CMD_START)

    def stop(self) -> dict | None:
        return self.command(p.CMD_STOP)

    def status(self) -> dict | None:
        return self.command(p.CMD_STATUS)

    def shutdown(self) -> dict | None:
        return self.command(p.CMD_SHUTDOWN)

    def listen_events(self, on_wake: Callable[[dict], None] | None = None) -> None:
        """阻塞接收并处理事件；``on_wake`` 仅在 wake 事件时回调。"""
        for msg in self.messages():
            if msg.get("type") == p.TYPE_WAKE and on_wake is not None:
                on_wake(msg)
            else:
                logger.debug("收到消息: %s", msg)
