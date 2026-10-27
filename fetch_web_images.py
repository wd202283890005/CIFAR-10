"""从 Wikimedia Commons 抓一批真实照片，用来测试训练好的权重。

Commons 的图片是自由版权（CC / 公有领域），适合拿来做演示；
本脚本会记录每张图的来源页面，方便追溯署名。

用法：
    python fetch_web_images.py                 # 每类 2 张，存到 data/web_images/
    python fetch_web_images.py --per-class 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.wikimedia.org/core/v1/commons"
UA = {"User-Agent": "cifar10-demo/1.0 (local model evaluation)"}

# CIFAR-10 的 10 个类，以及各自在 Commons 上比较好搜的关键词
CLASS_QUERIES = {
    "airplane": "airliner in flight",
    "automobile": "automobile car street",
    "bird": "bird perched",
    "cat": "domestic cat",
    "deer": "deer in field",
    "dog": "dog portrait",
    "frog": "frog",
    "horse": "horse standing",
    "ship": "cargo ship sea",
    "truck": "truck lorry",
}

OK_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _retry(fn, what: str, attempts: int = 5):
    """Commons 对匿名请求限制很严，429 和连接重置都很常见，统一退避重试。"""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 + 2.0 * i + random.random())
    raise RuntimeError(f"{what} 重试 {attempts} 次仍失败: {str(last)[:80]}")


def get_json(url: str) -> dict:
    def _do():
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    return _retry(_do, "请求接口")


def download(url: str, dest: Path) -> None:
    def _do():
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())

    _retry(_do, "下载图片")


def safe_name(title: str) -> str:
    """Commons 的标题可能很长且带特殊字符，压成安全文件名，并保留扩展名。"""
    path = Path(title.split(":", 1)[-1])
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", path.stem)[:50].strip("_")
    return f"{stem}{path.suffix.lower()}"


def fetch_class(cls: str, query: str, per_class: int, out_dir: Path) -> list[dict]:
    url = f"{API}/search/page?q={urllib.parse.quote('filetype:bitmap ' + query)}&limit=30"
    try:
        pages = get_json(url).get("pages") or []
    except Exception as exc:  # noqa: BLE001
        print(f"  搜索失败 {cls}: {exc}")
        return []

    saved: list[dict] = []
    for page in pages:
        if len(saved) >= per_class:
            break
        title = page.get("title", "")
        if not title.startswith("File:"):
            continue
        if Path(title).suffix.lower() not in OK_EXT:
            continue
        try:
            info = get_json(f"{API}/file/{urllib.parse.quote(title, safe='')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  取信息失败 {title}: {str(exc)[:60]}")
            continue

        cand = [info.get("preferred") or {}, info.get("original") or {}]
        dest = out_dir / cls / f"{len(saved) + 1}_{safe_name(title)}"
        dest.parent.mkdir(parents=True, exist_ok=True)

        got = False
        for variant in cand:
            src = variant.get("url")
            if not src:
                continue
            try:
                download(src, dest)
                got = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  下载失败 {title}: {str(exc)[:60]}")

        if not got or dest.stat().st_size < 4096:
            dest.unlink(missing_ok=True)
            continue

        saved.append(
            {
                "class": cls,
                "file": str(dest.relative_to(out_dir)),
                "title": title,
                "page": info.get("file_description_url"),
                "width": (info.get("original") or {}).get("width"),
                "height": (info.get("original") or {}).get("height"),
            }
        )
        print(f"  ✓ {cls:10s} {dest.name}  ({(info.get('original') or {}).get('width')}px)")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--out", default="data/web_images")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = out_dir / "sources.json"
    records: list[dict] = []
    if sources.exists():
        records = json.loads(sources.read_text(encoding="utf-8"))

    for cls, query in CLASS_QUERIES.items():
        have = [
            p
            for p in (out_dir / cls).glob("*")
            if p.is_file() and p.suffix.lower() in OK_EXT
        ] if (out_dir / cls).exists() else []
        if len(have) >= args.per_class:
            print(f"{cls}（已有 {len(have)} 张，跳过）")
            continue
        print(f"{cls}（搜索：{query}）")
        records.extend(fetch_class(cls, query, args.per_class - len(have), out_dir))
        time.sleep(1.0)

    sources.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    n_files = len([p for p in out_dir.rglob("*") if p.is_file() and p.suffix.lower() in OK_EXT])
    print(f"\n共 {n_files} 张图片，来源记录：{sources.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
