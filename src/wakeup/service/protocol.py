"""WebSocket 控制协议的消息常量与 JSON 编解码工具。

任何语言的程序都能连接 ``ws://127.0.0.1:8766/v1/wake/ws``，发送/接收 JSON 对象来集成。

客户端 -> 服务端（命令）::

    {"type": "start"}        开始监听
    {"type": "stop"}         停止监听（释放麦克风，省电）
    {"type": "status"}       查询状态
    {"type": "ping"}         连通性测试
    {"type": "shutdown"}     关闭整个服务

服务端 -> 客户端（响应 / 事件）::

    {"type": "ack",    "cmd": "...", "ok": true}
    {"type": "status", "listening": true/false, "model": "xiaoyuan", ...}
    {"type": "wake",   "model": "xiaoyuan", "score": 0.97, "ts": 1700000000.0}
    {"type": "error",  "message": "..."}

所有已连接的客户端都会收到 ``wake`` 事件广播。
"""

from __future__ import annotations

import json

# 命令
CMD_START = "start"
CMD_STOP = "stop"
CMD_STATUS = "status"
CMD_PING = "ping"
CMD_SHUTDOWN = "shutdown"

# 服务端消息类型
TYPE_ACK = "ack"
TYPE_STATUS = "status"
TYPE_WAKE = "wake"
TYPE_ERROR = "error"
TYPE_PONG = "pong"


def encode(message: dict) -> bytes:
    """dict -> UTF-8 JSON bytes."""
    return json.dumps(message, ensure_ascii=False).encode("utf-8")


def decode(line: bytes | str) -> dict:
    """UTF-8 JSON bytes/string -> dict."""
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    return json.loads(line)
