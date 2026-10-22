"""CIFAR-10 分类训练 —— 经典配方。

配方（CIFAR-10 上沿用多年的标准做法）：
    模型     ResNet-18 的 CIFAR 变体（3x3 stem、无 maxpool）
    增强     RandomCrop(32, padding=4) + RandomHorizontalFlip
    归一化   按 CIFAR-10 的通道均值/标准差
    优化器   SGD，momentum 0.9，weight decay 5e-4，初始 lr 0.1
    调度     阶梯退火，默认在 50% 和 75% 的 epoch 处各降 10 倍
    评估     每个 epoch 在测试集上算准确率，按最优准确率存权重

用法：
    python train.py                                  # 默认 200 epoch，ResNet-18
    python train.py --epochs 40 --model resnet18     # 短一些的跑法
    python train.py --model simple --epochs 10       # 快速验证流程
    python train.py --resume checkpoints/last.pt     # 断点续训
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from models import build_model, count_params

# CIFAR-10 全体训练集的通道统计量
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, list[str]]:
    root = Path(args.data_root)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )

    train_set = datasets.CIFAR10(root, train=True, download=False, transform=train_tf)
    test_set = datasets.CIFAR10(root, train=False, download=False, transform=test_tf)
    classes = train_set.classes

    if args.limit_train:
        train_set = Subset(train_set, range(min(args.limit_train, len(train_set))))
    if args.limit_test:
        test_set = Subset(test_set, range(min(args.limit_test, len(test_set))))

    common = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)

    classes = datasets.CIFAR10(root, train=True, download=False).classes
    return train_loader, test_loader, classes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        seen += targets.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        seen += targets.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)


def build_scheduler(args, optimizer):
    epochs = args.epochs
    if args.schedule == "step":
        fractions = [float(x) for x in args.milestones.split(",")]
        milestones = sorted({max(1, int(epochs * f)) for f in fractions})
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=args.gamma
        ), milestones
    if args.schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        ), []
    raise ValueError(f"未知调度 {args.schedule!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="CIFAR-10 经典配方训练")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet34", "simple"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--schedule", default="step", choices=["step", "cosine"])
    parser.add_argument("--milestones", default="0.5,0.75", help="lr 衰减点，按总 epoch 的比例")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="混合精度，CUDA 上默认开启")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--limit-train", type=int, default=0, help="只用前 N 张训练（调试用）")
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true", help="只评估 --resume 指定的权重")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp if args.amp is not None else device.type == "cuda"
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.logdir)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, classes = build_loaders(args)
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler, milestones = build_scheduler(args, optimizer)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and not args.eval_only:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_acc = ckpt.get("best_acc", 0.0)
        print(f"已加载 {args.resume}（epoch {ckpt.get('epoch', -1)}，best {best_acc:.2%}）")

    if args.eval_only:
        loss, acc = evaluate(model, test_loader, criterion, device, use_amp)
        print(f"测试集：loss {loss:.4f}  准确率 {acc:.2%}")
        return 0

    print(f"设备      : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"模型      : {args.model}  ({count_params(model):,} 参数)")
    print(f"训练样本  : {len(train_loader.dataset):,}   测试样本: {len(test_loader.dataset):,}")
    print(f"epoch     : {args.epochs}   batch: {args.batch_size}   混合精度: {use_amp}")
    print(f"调度      : {args.schedule}  milestones={milestones}  gamma={args.gamma}")
    print("-" * 88)

    csv_path = log_dir / f"{args.model}_log.csv"
    write_header = not csv_path.exists() or start_epoch == 0
    csv_file = csv_path.open("a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(
            ["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "seconds"]
        )
        csv_file.flush()

    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
    run_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, use_amp)
        scheduler.step()
        seconds = time.time() - t0

        print(
            f"epoch {epoch + 1:3d}/{args.epochs}  lr {lr:.4f}  "
            f"train {train_loss:.4f} / {train_acc:6.2%}  "
            f"test {test_loss:.4f} / {test_acc:6.2%}  {seconds:.1f}s"
        )

        csv_writer.writerow(
            [epoch + 1, f"{lr:.6f}", f"{train_loss:.6f}", f"{train_acc:.6f}",
             f"{test_loss:.6f}", f"{test_acc:.6f}", f"{seconds:.2f}"]
        )
        csv_file.flush()
        writer.add_scalar("loss/train", train_loss, epoch + 1)
        writer.add_scalar("loss/test", test_loss, epoch + 1)
        writer.add_scalar("acc/train", train_acc, epoch + 1)
        writer.add_scalar("acc/test", test_acc, epoch + 1)
        writer.add_scalar("lr", lr, epoch + 1)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "args": vars(args),
        }
        torch.save(state, out_dir / "last.pt")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(state, out_dir / "best.pt")
            print(f"           ↑ 新的最佳准确率 {best_acc:.2%}，已保存 best.pt")

    csv_file.close()
    writer.close()

    total = time.time() - run_start
    print("-" * 88)
    print(f"训练结束，用时 {total / 60:.1f} 分钟，最佳测试准确率 {best_acc:.2%}")

    summary = {
        "model": args.model,
        "epochs": args.epochs,
        "best_test_acc": best_acc,
        "minutes": round(total / 60, 2),
        "params": count_params(model),
    }
    (log_dir / f"{args.model}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
