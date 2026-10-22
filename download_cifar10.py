"""并行下载并解压 CIFAR-10 数据集。

torchvision 自带的下载走单连接，从多伦多大学服务器只有几十 KB/s（一个多小时），
这里改用 HTTP Range 多线程分块下载，速度约提升 4 倍，并校验 MD5。

注意：服务器对并发有上限，线程开太多会被直接断连，8 线程是比较稳的取值；
此外该校服务器在国内网络下时常整段连不上，遇到就隔一会儿再跑。

用法：
    python download_cifar10.py [--workers 8] [--out data]

下载完成后，train.py 里的 datasets.CIFAR10(root='data', download=True)
会直接发现本地文件，不再联网。
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URLS = [
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz",
]
FILENAME = "cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"
EXTRACTED = "cifar-10-batches-py"
CHUNK = 1024 * 1024
RETRIES = 8


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int) -> bytes:
    """取 [start, end] 闭区间的字节，失败时按缺多少补多少。"""
    buf = bytearray()
    attempt = 0
    while len(buf) < end - start + 1:
        want_start = start + len(buf)
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={want_start}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=90) as r:
                chunk = r.read()
            if not chunk:
                raise IOError("空响应")
            buf.extend(chunk)
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if attempt > RETRIES:
                raise RuntimeError(
                    f"分块 {start}-{end} 重试 {RETRIES} 次仍失败: {exc}"
                ) from exc
            time.sleep(min(1.0 + attempt * 1.5, 8.0) + random.random())
    return bytes(buf)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(tar_path: Path, workers: int) -> bool:
    url = None
    total = 0
    for candidate in URLS:
        try:
            total = head_size(candidate)
            url = candidate
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  连接失败 {candidate}: {str(exc)[:80]}")
    if url is None:
        print("所有下载地址都不可用", file=sys.stderr)
        return False

    print(f"下载 {url}")
    print(f"大小 {total / 1024**2:.1f} MB，{workers} 线程分块下载")

    ranges = [
        (start, min(start + CHUNK - 1, total - 1)) for start in range(0, total, CHUNK)
    ]
    done = 0
    lock = threading.Lock()
    started = time.time()
    failed: list[str] = []

    with tar_path.open("wb") as f:
        f.truncate(total)

    def task(idx: int) -> None:
        nonlocal done
        start, end = ranges[idx]
        try:
            data = fetch_range(url, start, end)
        except Exception as exc:  # noqa: BLE001
            failed.append(str(exc))
            return
        with lock:
            with tar_path.open("r+b") as f:
                f.seek(start)
                f.write(data)
            done += 1
            moved = done * CHUNK / 1024**2
            speed = moved / max(time.time() - started, 1e-6)
            sys.stdout.write(
                f"\r  {done}/{len(ranges)} 块  {moved:.0f} MB  {speed:.2f} MB/s"
            )
            sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in [pool.submit(task, i) for i in range(len(ranges))]:
            fut.result()
    print()

    if failed:
        print(f"{len(failed)} 个分块失败：{failed[0]}", file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / FILENAME
    extracted = out_dir / EXTRACTED

    if extracted.is_dir() and (extracted / "data_batch_1").exists():
        print(f"已存在解压后的数据集：{extracted}")
        return 0

    # 下载 → 校验，最多尝试两轮（服务器偶尔会传坏数据）
    for attempt in range(1, 3):
        if not tar_path.exists() or tar_path.stat().st_size == 0:
            if not download(tar_path, args.workers):
                return 1
        print("校验 MD5 ...")
        actual = md5sum(tar_path)
        if actual == MD5:
            print(f"  MD5 OK ({actual})")
            break
        tar_path.unlink()
        if attempt == 2:
            print(f"MD5 两次都不匹配（{actual}），请稍后重试", file=sys.stderr)
            return 1
        print(f"  MD5 不匹配（{actual}），删除后重新下载", file=sys.stderr)

    print("解压 ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(out_dir, filter="data")
    print(f"完成：{extracted}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中断，重新运行会从头下载", file=sys.stderr)
        raise SystemExit(130)
