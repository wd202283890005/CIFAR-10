"""用训练好的权重给任意图片分类。

权重怎么用，本质上就四步：
    1. 读 checkpoint（里面存了 state_dict，还有训练时的参数，包括模型名）
    2. 用同样的模型结构实例化，再 load_state_dict
    3. 把图片预处理成训练时的样子：RGB -> 缩放 -> ToTensor -> 按 CIFAR-10 均值方差归一化
    4. forward 取 argmax（softmax 得到置信度）

用法：
    python infer.py --input some.jpg
    python infer.py --input data/web_images        # 整个目录（会递归）
    python infer.py --input data/web_images --out data/web_test.png

注意：模型是在 32x32 的小图上训练的，任何输入都会被压到 32x32。
大图、构图复杂的照片会明显掉准确率，这是 CIFAR-10 模型的固有限制，不是代码问题。
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import build_model

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

for _name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
    if any(f.name == _name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def build_transform(mode: str) -> transforms.Compose:
    if mode == "stretch":
        geom = [transforms.Resize((32, 32))]
    else:  # center：短边缩到 32，再中心裁剪，保持长宽比
        geom = [transforms.Resize(32), transforms.CenterCrop(32)]
    return transforms.Compose(geom + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def collect_images(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            paths.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXT))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"跳过（不存在）：{t}")
    return paths


def load_model(ckpt_path: str, device: torch.device) -> tuple[torch.nn.Module, dict, list[str]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    model_name = saved.get("model", "resnet18")
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    return model, ckpt, classes


@torch.no_grad()
def predict_paths(
    model: torch.nn.Module, paths: list[Path], tf, device: torch.device, batch: int = 32
) -> tuple[torch.Tensor, list[np.ndarray]]:
    probs_all: list[torch.Tensor] = []
    shown: list[np.ndarray] = []
    for i in range(0, len(paths), batch):
        chunk = paths[i : i + batch]
        tensors, arrays = [], []
        for p in chunk:
            with Image.open(p) as im:
                im = im.convert("RGB")
                t = tf(im)
            tensors.append(t)
            # 显示用的是模型真正看到的 32x32 图（反归一化后）
            vis = (t * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1))
            arrays.append(vis.clamp(0, 1).permute(1, 2, 0).numpy())
        logits = model(torch.stack(tensors).to(device))
        probs_all.append(F.softmax(logits, dim=1).cpu())
        shown.extend(arrays)
    return torch.cat(probs_all), shown


def guess_truth(path: Path, classes: list[str]) -> str | None:
    """如果图片放在以类名命名的目录里，就直接当成真值。"""
    for part in reversed(path.parts[:-1]):
        if part.lower() in classes:
            return part.lower()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="用 CIFAR-10 权重给任意图片分类")
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--input", nargs="+", default=["data/web_images"])
    parser.add_argument("--mode", choices=["center", "stretch"], default="center")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--out", default="data/web_test.png", help="结果图，留空则不画")
    parser.add_argument("--cols", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, classes = load_model(args.ckpt, device)
    paths = collect_images(args.input)
    if not paths:
        print("没找到图片")
        return 1

    print(f"权重   : {args.ckpt}（第 {ckpt.get('epoch', -1) + 1} 个 epoch）")
    print(f"设备   : {device}    图片数: {len(paths)}    预处理: {args.mode}")
    print("-" * 92)
    print(f"{'图片':38s} {'预测':12s} {'置信度':>7s}   真值     结果")

    tf = build_transform(args.mode)
    probs, shown = predict_paths(model, paths, tf, device)
    top_probs, top_idx = probs.topk(args.topk, dim=1)

    correct = judged = 0
    for i, path in enumerate(paths):
        pred = classes[top_idx[i, 0]]
        conf = top_probs[i, 0].item()
        truth = guess_truth(path, classes)
        mark = ""
        if truth:
            judged += 1
            ok = pred == truth
            correct += ok
            mark = "✓" if ok else f"✗ ({pred} ← {truth})"
        name = str(path.name)[:36]
        print(f"{name:38s} {pred:12s} {conf:7.1%}   {truth or '-':8s} {mark}")

    if judged:
        print("-" * 92)
        print(f"已知真值的 {judged} 张中，正确 {correct} 张，准确率 {correct / judged:.0%}")

    print("\n每张图的前 3 个候选：")
    for i, path in enumerate(paths):
        cand = ", ".join(
            f"{classes[j]} {p:.0%}" for j, p in zip(top_idx[i].tolist(), top_probs[i].tolist())
        )
        print(f"  {path.name[:40]:42s} {cand}")

    if args.out:
        n = len(paths)
        cols = min(args.cols, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.7), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for i, path in enumerate(paths):
            ax = axes[i // cols][i % cols]
            ax.imshow(shown[i], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            pred = classes[top_idx[i, 0]]
            truth = guess_truth(path, classes)
            color = "#2e9e4f"
            if truth and pred != truth:
                color = "#d93a3a"
            ax.set_title(f"{pred} {top_probs[i, 0]:.0%}", fontsize=10, color=color, pad=3)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2.5)
            if truth:
                ax.text(
                    0.5, -0.06, f"真值 {truth}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=8,
                    color="#555" if pred == truth else "#d93a3a",
                )
        acc = f"{correct / judged:.0%}" if judged else "n/a"
        fig.suptitle(
            f"网络图片测试（模型看到的 32×32 输入，绿=正确 红=错误，已知真值准确率 {acc}）",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        print(f"\n结果图：{out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
