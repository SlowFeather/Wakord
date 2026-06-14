"""下载工具：带进度条、支持断点续传到临时文件后原子改名。"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from tqdm import tqdm

from .._logging import get_logger

logger = get_logger(__name__)


def download(url: str, dest: Path, desc: str | None = None) -> Path:
    """下载 ``url`` 到 ``dest``；若文件已存在则跳过。"""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("已存在，跳过下载: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    desc = desc or dest.name

    logger.info("下载 %s -> %s", url, dest)
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as bar:
        def _hook(block_num, block_size, total_size):
            if total_size > 0:
                bar.total = total_size
            bar.update(block_size)

        urllib.request.urlretrieve(url, tmp, reporthook=_hook)

    tmp.replace(dest)
    return dest
