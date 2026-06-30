# UV 操作备忘

这个文档是本项目本地使用 `uv` 的操作备忘，不随代码提交。

## 首次创建环境

```powershell
uv sync --extra dev
```

这会按 `uv.lock` 创建 `.venv`，并安装默认运行依赖和测试依赖。

如果要一次性安装训练、Edge TTS、VAD 等完整依赖：

```powershell
uv sync --extra all
```

## 严格按锁文件安装

```powershell
uv sync --frozen --extra dev
uv sync --frozen --extra all
```

`--frozen` 表示不重新解析依赖，只使用现有 `uv.lock`。换机器、CI 或复现环境时优先使用。

## 运行命令

```powershell
uv run wakeup --help
uv run wakeup prepare
uv run wakeup fit
uv run wakeup listen --show-score
uv run wakeup serve --listen
```

## 跑测试

```powershell
uv run --frozen --extra dev python -m compileall -q src tests main.py
uv run --frozen --extra dev python -m pytest -q
```

## 更新锁文件

修改 `pyproject.toml` 的依赖后运行：

```powershell
uv lock
```

如果希望同步环境：

```powershell
uv sync --extra dev
```

## 查看环境信息

```powershell
uv run python --version
uv run python -c "import sys; print(sys.executable)"
uv pip list
```

## 清理环境

```powershell
Remove-Item -Recurse -Force .venv
uv sync --extra dev
```

PowerShell 删除 `.venv` 前确认当前目录是项目根目录。
