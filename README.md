# CIFAR-10 分类训练

本项目使用 PyTorch 在 CIFAR-10 上训练图像分类模型。虚拟环境由 conda 创建，直接放在项目文件夹内（`.conda/cifar10`），整个项目因此可以整体拷贝到别的机器。

## 环境信息

| 项目 | 值 |
| --- | --- |
| 环境路径 | `.conda/cifar10`（项目内，前缀环境） |
| Python | 3.11.17 |
| PyTorch | 2.5.1 + cu121 |
| torchvision | 0.20.1 + cu121 |
| cuDNN | 9.1.0 |
| GPU | NVIDIA GeForce RTX 3060 Laptop（6 GB，驱动 546.30，compute capability 8.6） |
| pip 索引 | 清华源（已写入环境内 `pip.ini`） |

## 激活环境

conda 前缀环境不能只写名字激活，需要给完整路径：

```powershell
conda activate "D:\programing Python\Python  document\CIFAR-10\.conda\cifar10"
```

嫌路径太长就双击 `env.bat`，它会打开一个已激活本环境的命令行窗口。

也可以完全绕开 conda，直接用环境里的解释器：

```powershell
& "D:\programing Python\Python  document\CIFAR-10\.conda\cifar10\python.exe" check_env.py
```

## 自检

```powershell
python check_env.py
```

打印 Python / torch / CUDA / 显存信息，跑一次 fp16 矩阵乘测速，并用随机数据跑通 5 步训练循环。全部通过即表示环境可用。本机实测：cuDNN 可用，fp16 约 8.3 TFLOPS，峰值显存 109 MB。

## 数据集

```powershell
python download_cifar10.py
```

从多伦多大学服务器并行分块下载 `cifar-10-python.tar.gz`，校验 MD5 后解压到 `data/cifar-10-batches-py`。官方单连接下载在这里只有几十 KB/s（要一个多小时），这个脚本用 8 线程能快 4 倍左右。

两个现实问题：

- 线程开到 16 以上会被服务器直接断连，默认的 8 线程别再往上加；
- 这两台服务器在国内网络下时常整段连不上（连接超时），遇到就过一会儿重跑。

数据齐了以后，`datasets.CIFAR10(root="data", download=True)` 会直接复用本地文件，不再联网。

数据集是 pickle 文件不是图片，想肉眼看就渲染一张：

```powershell
python preview_data.py            # 每类抽 10 张，输出 data/preview.png
```

## 训练

采用 CIFAR-10 上沿用多年的经典配方：

| 环节 | 设置 |
| --- | --- |
| 模型 | ResNet-18 的 CIFAR 变体（3×3 stem、去掉 maxpool），约 1115 万参数 |
| 数据增强 | `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` |
| 归一化 | 通道均值 `(0.4914, 0.4822, 0.4465)`，标准差 `(0.2470, 0.2435, 0.2616)` |
| 优化器 | SGD，momentum 0.9，weight decay 5e-4，初始 lr 0.1 |
| 学习率调度 | 阶梯退火，在 50% 和 75% 的 epoch 处各降 10 倍 |
| 精度 | CUDA 上自动启用混合精度（`--no-amp` 可关） |

```powershell
python train.py                                   # 完整 200 epoch
python train.py --epochs 40 --workers 4           # 快速跑一版
python train.py --model simple --epochs 10        # 两层 CNN，验证流程
python train.py --resume checkpoints/last.pt      # 断点续训
python train.py --resume checkpoints/best.pt --eval-only   # 只评估
```

训练过程每个 epoch 输出一行，权重和指标分别落到：

- `checkpoints/best.pt`、`checkpoints/last.pt`
- `logs/<模型>_log.csv`（每轮 loss/acc/耗时）
- `logs/tensorboard/`（`tensorboard --logdir logs/tensorboard` 可视化）

本机实测（RTX 3060 Laptop，40 epoch，`--workers 0`）：约 28 秒/轮，总耗时约 19 分钟，测试集准确率 93% 左右。用 `--workers 4` 会更快。

## 预测

```powershell
python predict.py                       # 用 checkpoints/best.pt
python predict.py --samples 200         # 多看几张
```

输出整体准确率、各类准确率，并把预测结果画成 `data/predictions.png`（绿框正确、红框错误并标出真值）。

## 依赖

- `requirements.txt`：pip 侧完整依赖（由 `pip freeze` 导出）
- 新增依赖：`python -m pip install <包名>`（自动走清华源）

PyTorch 的 CUDA 版必须从官方源装，普通镜像里是 CPU 版：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 目录

```
CIFAR-10/
├── .conda/               # 虚拟环境（不入库）
│   └── cifar10/
├── data/                 # 数据集与生成的可视化图片（不入库）
├── checkpoints/          # 权重（不入库）
├── logs/                 # 训练日志与 TensorBoard（不入库）
├── .gitignore
├── env.bat               # 打开已激活环境的命令行
├── check_env.py          # 环境自检
├── preview_data.py       # 数据集预览图
├── download_cifar10.py   # 数据集并行下载
├── models.py             # ResNet-18/34 与简单 CNN
├── train.py              # 训练
├── predict.py            # 评估与预测可视化
├── requirements.txt
└── README.md
```

## 注意

- 路径里含空格，命令中的路径一律加引号。
- 若某些工具仍被空格路径卡住，可临时映射短路径：`subst X: "D:\programing Python\Python  document\CIFAR-10"`，用完 `subst X: /d` 解除。
- `numpy` 是 2.4.6。torch 2.5.1 对 numpy 2.x 兼容，自检已实际验证过，无需降级。
