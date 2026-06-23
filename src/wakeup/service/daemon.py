"""Small process supervisor helpers for running ``wakeup serve`` in the background."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import Config
from .._logging import get_logger
from .client import ServiceClient

logger = get_logger(__name__)


@dataclass
class DaemonPaths:
    run_dir: Path
    pid_file: Path
    log_file: Path
    meta_file: Path


def daemon_paths(cfg: Config) -> DaemonPaths:
    run_dir = cfg.fs.base / "run"
    return DaemonPaths(
        run_dir=run_dir,
        pid_file=run_dir / "wakeup.pid",
        log_file=run_dir / "wakeup-service.log",
        meta_file=run_dir / "wakeup-daemon.json",
    )


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _service_responds(cfg: Config, timeout: float = 0.5) -> bool:
    try:
        with ServiceClient(cfg.service.host, cfg.service.port, timeout=timeout) as client:
            resp = client.status()
        return bool(resp and resp.get("type") == "status")
    except Exception:
        return False


def status_daemon(cfg: Config) -> dict:
    paths = daemon_paths(cfg)
    pid = _read_pid(paths.pid_file)
    return {
        "pid": pid,
        "process_alive": _pid_alive(pid),
        "service_responding": _service_responds(cfg),
        "pid_file": str(paths.pid_file),
        "log_file": str(paths.log_file),
    }


def start_daemon(
    cfg: Config,
    *,
    config_path: str | None = None,
    listen: bool = False,
    extra_args: list[str] | None = None,
    wait_seconds: float = 10.0,
) -> dict:
    paths = daemon_paths(cfg)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    current = status_daemon(cfg)
    if current["process_alive"] or current["service_responding"]:
        current["started"] = False
        current["message"] = "daemon already appears to be running"
        return current

    cmd = [sys.executable, "-m", "wakeup.cli", "serve"]
    if config_path:
        cmd.extend(["--config", config_path])
    if listen:
        cmd.append("--listen")
    if extra_args:
        cmd.extend(extra_args)

    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        start_new_session = True

    log = paths.log_file.open("ab")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=(os.name != "nt"),
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
    finally:
        log.close()

    paths.pid_file.write_text(str(proc.pid), encoding="utf-8")
    paths.meta_file.write_text(
        json.dumps(
            {
                "pid": proc.pid,
                "cmd": cmd,
                "cwd": str(Path.cwd()),
                "log_file": str(paths.log_file),
                "started_at": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _service_responds(cfg):
            break
        if proc.poll() is not None:
            break
        time.sleep(0.25)

    result = status_daemon(cfg)
    result["started"] = True
    result["cmd"] = cmd
    return result


def stop_daemon(cfg: Config, *, timeout: float = 10.0) -> dict:
    paths = daemon_paths(cfg)
    pid = _read_pid(paths.pid_file)

    try:
        with ServiceClient(cfg.service.host, cfg.service.port, timeout=2.0) as client:
            client.shutdown()
    except Exception as exc:
        logger.debug("service shutdown command failed: %s", exc)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _service_responds(cfg) and not _pid_alive(pid):
            break
        time.sleep(0.25)

    if _pid_alive(pid):
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    for path in (paths.pid_file, paths.meta_file):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    result = status_daemon(cfg)
    result["stopped"] = True
    return result


def install_autostart(cfg: Config, *, config_path: str | None = None, listen: bool = False) -> dict:
    """Install a user-level autostart entry for the current platform."""
    paths = daemon_paths(cfg)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "wakeup.cli", "serve"]
    if config_path:
        cmd.extend(["--config", str(Path(config_path).resolve())])
    if listen:
        cmd.append("--listen")

    if os.name == "nt":
        task_name = "WakeUpVoiceService"
        ps_cmd = " ".join([f'"{c}"' if " " in c else c for c in cmd])
        action = f"cmd /c cd /d {Path.cwd()} && {ps_cmd} >> {paths.log_file} 2>&1"
        subprocess.run(
            [
                "schtasks",
                "/Create",
                "/TN",
                task_name,
                "/SC",
                "ONLOGON",
                "/TR",
                action,
                "/F",
            ],
            check=True,
        )
        return {"installed": True, "kind": "schtasks", "name": task_name, "command": action}

    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit = unit_dir / "wakeup.service"
    unit.write_text(
        "\n".join(
            [
                "[Unit]",
                "Description=WakeUp voice wake-word service",
                "",
                "[Service]",
                f"WorkingDirectory={Path.cwd()}",
                "ExecStart=" + " ".join(cmd),
                "Restart=on-failure",
                "RestartSec=3",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "--now", "wakeup.service"], check=True)
    return {"installed": True, "kind": "systemd-user", "unit": str(unit)}


def uninstall_autostart() -> dict:
    if os.name == "nt":
        task_name = "WakeUpVoiceService"
        subprocess.run(["schtasks", "/Delete", "/TN", task_name, "/F"], check=False)
        return {"uninstalled": True, "kind": "schtasks", "name": task_name}

    subprocess.run(["systemctl", "--user", "disable", "--now", "wakeup.service"], check=False)
    unit = Path.home() / ".config" / "systemd" / "user" / "wakeup.service"
    try:
        unit.unlink()
    except FileNotFoundError:
        pass
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    return {"uninstalled": True, "kind": "systemd-user", "unit": str(unit)}


def as_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    try:
        return asdict(value)
    except TypeError:
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
