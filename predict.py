"""用训练好的权重在测试集上评估，并把预测结果画成图。

用法：
    python predict.py                                  # 用 checkpoints/best.pt
    python predict.py --ckpt checkpoints/last.pt --samples 100
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".conda" / "mpl-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_model

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

for _name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
    if any(f.name == _name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """(3,H,W) 归一化张量 -> (H,W,3) 的 0~1 数组，用于显示。"""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="data/predictions.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    model_name = saved.get("model", "resnet18")
    base = Path(args.data_root) / "cifar-10-batches-py"
    if not base.exists():
        raise SystemExit(f"找不到数据集：{base}")

    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_set = datasets.CIFAR10(
        args.data_root,
        train=False,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    )
    loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            all_preds.append(logits.argmax(1).cpu())
            all_targets.append(targets)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    acc = (preds == targets).float().mean().item()
    print(f"权重    : {args.ckpt}（第 {ckpt.get('epoch', -1) + 1} 个 epoch）")
    print(f"模型    : {model_name}   测试集 {len(targets)} 张")
    print(f"准确率  : {acc:.2%}")
    print("各类准确率：")
    per_class = []
    for i, name in enumerate(test_set.classes):
        mask = targets == i
        a = (preds[mask] == targets[mask]).float().mean().item()
        per_class.append((name, a, int(mask.sum())))
        print(f"  {name:10s} {a:6.2%}  ({mask.sum().item()} 张)")
    worst = min(per_class, key=lambda x: x[1])
    print(f"最弱类别: {worst[0]} ({worst[1]:.2%})")

    # 画一张预测图：正确绿色边框，错误红色边框并标出真实类别
    rng = np.random.default_rng(args.seed)
    n = args.samples
    picks = rng.choice(len(test_set), size=min(n, len(test_set)), replace=False)
    cols = 10
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.0), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, sample in enumerate(picks):
        img_tensor, label = test_set[int(sample)]
        pred = int(preds[sample])
        ax = axes[idx // cols][idx % cols]
        ax.imshow(denormalize(img_tensor))
        ax.set_xticks([])
        ax.set_yticks([])
        ok = pred == label
        color = "#2e9e4f" if ok else "#d93a3a"
        ax.set_title(
            f"{test_set.classes[pred]}",
            fontsize=9,
            color=color,
            pad=2,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2.5)
        if not ok:
            ax.text(
                0.5, -0.10, f"真值 {test_set.classes[label]}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7, color="#d93a3a",
            )

    fig.suptitle(f"{model_name} 测试集预测（整体准确率 {acc:.2%}，红框为错误）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"已保存  : {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
