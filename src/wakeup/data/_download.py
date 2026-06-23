"""Download helpers with resume, retry, and optional SHA256 verification."""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

from .._logging import get_logger

logger = get_logger(__name__)

_CHUNK = 256 * 1024
_UA = {"User-Agent": "wakeup/0.1 (+https://github.com)"}


def sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest of a local file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: Path, expected: str | None) -> None:
    """Validate a file checksum, deleting the bad file before raising."""
    if not expected:
        return
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        try:
            Path(path).unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected}, got {actual}. "
            "The file was removed; retry the download or update the configured checksum."
        )


def download(
    url: str,
    dest: Path,
    desc: str | None = None,
    retries: int = 8,
    timeout: float = 60.0,
    sha256: str | None = None,
) -> Path:
    """Download ``url`` to ``dest``; resume partial files and optionally verify SHA256."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        verify_sha256(dest, sha256)
        logger.info("Existing file verified, skipping download: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    desc = desc or dest.name

    logger.info("Downloading %s -> %s", url, dest)
    for attempt in range(1, retries + 1):
        existing = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers=dict(_UA))
        if existing:
            req.add_header("Range", f"bytes={existing}-")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", resp.getcode())
                if status == 206:
                    cr = resp.headers.get("Content-Range", "")
                    total = int(cr.split("/")[-1]) if "/" in cr else None
                    mode = "ab"
                else:
                    cl = resp.headers.get("Content-Length")
                    total = int(cl) if cl else None
                    existing, mode = 0, "wb"

                with tmp.open(mode) as f, tqdm(
                    total=total,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc,
                ) as bar:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))

            size = tmp.stat().st_size
            if total is None or size >= total:
                tmp.replace(dest)
                verify_sha256(dest, sha256)
                logger.info("Download complete: %s (%d bytes)", dest, size)
                return dest
            logger.warning("Incomplete download %d/%d; retry %d/%d", size, total, attempt, retries)

        except urllib.error.HTTPError as exc:
            if exc.code == 416 and tmp.exists():
                tmp.replace(dest)
                verify_sha256(dest, sha256)
                logger.info("Download complete: %s", dest)
                return dest
            logger.warning("HTTP error %s; retry %d/%d", exc, attempt, retries)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            logger.warning("Download interrupted (%s); retry %d/%d", exc, attempt, retries)

        time.sleep(min(2 ** attempt, 15))

    raise RuntimeError(f"Download failed after {retries} retries: {url}")
