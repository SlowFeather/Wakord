"""控制协议：本机 TCP 上的换行分隔 JSON（JSON-lines）。

任何语言的程序都能用「连上 TCP、按行发 JSON、按行读 JSON」的方式集成。

客户端 -> 服务端（命令）::

    {"cmd": "start"}        开始监听
    {"cmd": "stop"}         停止监听（释放麦克风，省电）
    {"cmd": "status"}       查询状态
    {"cmd": "ping"}         连通性测试
    {"cmd": "shutdown"}     关闭整个服务

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
    """dict -> 一行 JSON（含换行）。"""
    return (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")


def decode(line: bytes | str) -> dict:
    """一行 JSON -> dict。"""
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    return json.loads(line)
