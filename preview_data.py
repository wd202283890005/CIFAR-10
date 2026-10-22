"""把 CIFAR-10 的 pickle 数据渲染成一张图片，方便肉眼检查。

为什么看不到图片文件：
    CIFAR-10 的 Python 版本不含任何 .png/.jpg。1200 多万张图挤在 7 个
    无扩展名的文件里，每个文件是一个 pickle，里面是 numpy 数组：
        data   : (10000, 3072) uint8   每行 = 一张 32x32 彩色图
        labels : 10000 个 0~9 的整数
        3072   : 32*32*3，通道顺序是 RRR...GGG...BBB（不是常见的 HWC）
    所以要用 reshape(3, 32, 32) 再转成 (32, 32, 3) 才能变成图像。

用法：
    python preview_data.py                      # 训练集，每类 10 张
    python preview_data.py --split test         # 测试集
    python preview_data.py --per-class 5 --out data/small.png
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

# matplotlib 默认把字体缓存写到用户目录，受限环境里可能没有写权限，
# 指到项目内，顺便让脚本自带缓存。
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".conda" / "mpl-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# 默认字体没有中文字形，标题会变成方框，这里找一个可用的中文字体
for _name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
    if any(f.name == _name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
TRAIN_BATCHES = [f"data_batch_{i}" for i in range(1, 6)]


def load_batches(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    names = TRAIN_BATCHES if split == "train" else ["test_batch"]
    data, labels = [], []
    for name in names:
        path = root / name
        if not path.exists():
            raise SystemExit(f"找不到 {path}，请先把数据集下载解压到 {root}")
        with path.open("rb") as f:
            batch = pickle.load(f, encoding="bytes")
        data.append(np.asarray(batch[b"data"], dtype=np.uint8))
        labels.append(np.asarray(batch[b"labels"], dtype=np.int64))
    return np.concatenate(data), np.concatenate(labels)


def decode(flat_row: np.ndarray) -> np.ndarray:
    """(3072,) uint8 -> (32, 32, 3) uint8"""
    return flat_row.reshape(3, 32, 32).transpose(1, 2, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/cifar-10-batches-py")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--out", default="data/preview.png")
    args = parser.parse_args()

    root = Path(args.root)
    data, labels = load_batches(root, args.split)
    print(f"{args.split}: {data.shape[0]} 张，{data.shape[1]} 维/张")

    by_class = {c: np.flatnonzero(labels == i) for i, c in enumerate(CLASSES)}
    for i, c in enumerate(CLASSES):
        print(f"  [{i}] {c:10s} {len(by_class[c]):5d} 张")

    n = args.per_class
    fig, axes = plt.subplots(
        len(CLASSES), n, figsize=(n * 1.5, len(CLASSES) * 1.6), squeeze=False
    )
    rng = np.random.default_rng(0)
    for row, cname in enumerate(CLASSES):
        picks = rng.choice(by_class[cname], size=n, replace=False)
        for col, idx in enumerate(picks):
            ax = axes[row][col]
            ax.imshow(decode(data[idx]))
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(cname, rotation=0, ha="right", va="center", fontsize=11)
    fig.suptitle(f"CIFAR-10 {args.split} 集预览（每类 {n} 张，共 {len(CLASSES) * n} 张）")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"已保存：{out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
