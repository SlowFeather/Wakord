"""示例：用另一个 Python 程序通过 WebSocket 控制唤醒词服务，并接收唤醒回调。

前提：已在另一个终端启动服务：
    wakeup serve

运行本示例：
    python examples/control_service.py
"""

from __future__ import annotations

import asyncio

from wakeup.service.ws_client import WsServiceClient, wake_ws_url


def on_wake(msg: dict) -> None:
    print(f"\n[回调] 检测到唤醒词！score={msg['score']}，可以在这里触发你的业务逻辑。")


async def main() -> None:
    url = wake_ws_url("127.0.0.1", 8766, "/v1/wake/ws")
    async with WsServiceClient(url) as client:
        print("发送 start：开始监听")
        print(await client.start())

        print("监听中，请说「小元」；按 Ctrl+C 退出。")
        async for msg in client.messages():
            if msg.get("type") == "wake":
                on_wake(msg)
            else:
                print(msg)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
