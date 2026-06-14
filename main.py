"""[已废弃] 旧的一体式 Colab 脚本已被工程化重构取代。

原来的 main.py 是为 Google Colab 写的（包含 !pip / !wget 等 notebook 魔法命令，
无法在本地直接运行）。其逻辑已拆分进 `src/wakeup/` 下的各模块，并提供了
`wakeup` 命令行。请改用：

    pip install -e .
    wakeup train      # 训练（合成中文正样本 → 特征 → 训练 → 导出 ONNX）
    wakeup serve      # 启动常驻监听服务
    wakeup ctl start  # 开始监听
    wakeup listen     # 前台调阈值

详见 README.md。本文件保留仅作迁移提示，确认无碍后可删除。
"""

import sys


def main() -> int:
    print(__doc__)
    print("提示：直接运行 `wakeup train` 开始训练，或 `wakeup --help` 查看全部命令。")
    # 方便起见，转发到新的 CLI（若已 pip install -e .）
    try:
        from wakeup.cli import main as cli_main
    except ImportError:
        print("\n尚未安装本项目，请先执行： pip install -e .")
        return 1
    if len(sys.argv) > 1:
        return cli_main(sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
