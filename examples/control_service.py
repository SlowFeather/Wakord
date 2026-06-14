"""示例：用另一个 Python 程序控制唤醒词服务，并接收唤醒回调。

前提：已在另一个终端启动服务 ——
    wakeup serve

运行本示例：
    python examples/control_service.py
"""

import threading
import time

from wakeup.service.client import ServiceClient


def on_wake(msg: dict) -> None:
    print(f"\n[回调] 检测到唤醒词！score={msg['score']}，可以在这里触发你的业务逻辑。")


def main() -> None:
    client = ServiceClient(host="127.0.0.1", port=8765).connect()

    # 后台线程持续接收事件（含唤醒广播）
    threading.Thread(
        target=client.listen_events, kwargs={"on_wake": on_wake}, daemon=True
    ).start()

    print("发送 start：开始监听")
    print(ServiceClient(port=8765).connect().start())

    print("监听 15 秒，请说「小元」...")
    time.sleep(15)

    print("发送 stop：停止监听（释放麦克风、省电）")
    with ServiceClient(port=8765) as c:
        print(c.stop())

    time.sleep(1)
    client.close()


if __name__ == "__main__":
    main()
