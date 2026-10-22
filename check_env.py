"""环境自检：确认 PyTorch / GPU / 依赖可用，并跑通一次最小训练循环。

用法：
    python check_env.py
"""

import platform
import sys
import time


def line(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    line("Python")
    print(f"version : {sys.version.split()[0]}")
    print(f"exec    : {sys.executable}")
    print(f"platform: {platform.platform()}")

    line("核心依赖")
    import numpy as np
    print(f"numpy        : {np.__version__}")

    import torch
    import torchvision

    print(f"torch        : {torch.__version__}  (built with CUDA {torch.version.cuda})")
    print(f"torchvision  : {torchvision.__version__}")
    print(f"cudnn        : {torch.backends.cudnn.version()}  enabled={torch.backends.cudnn.is_available()}")

    line("GPU")
    if not torch.cuda.is_available():
        print("CUDA 不可用 —— 将只能使用 CPU 训练。")
        print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        return 1

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(
            f"[{i}] {props.name} | 显存 {total / 1024**3:.1f} GB "
            f"(空闲 {free / 1024**3:.1f} GB) | compute capability {props.major}.{props.minor}"
        )
    device = torch.device("cuda:0")

    line("GPU 算力小测")
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tflops = 20 * 2 * 4096**3 / elapsed / 1e12
    print(f"fp16 矩阵乘 4096^3 x20: {elapsed:.2f}s -> 约 {tflops:.1f} TFLOPS")

    line("最小训练循环（随机数据，不下载数据集）")
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 10),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.randn(64, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (64,), device=device)

    model.train()
    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        print(f"step {step + 1}/5  loss={loss.item():.4f}")
    print(f"峰值显存: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

    line("结论")
    print("环境可用：torch + CUDA + 训练循环均正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
