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
├── data/                 # 数据集（不入库）
├── .gitignore
├── env.bat               # 打开已激活环境的命令行
├── check_env.py          # 环境自检
├── download_cifar10.py   # 数据集并行下载
├── requirements.txt
└── README.md
```

## 注意

- 路径里含空格，命令中的路径一律加引号。
- 若某些工具仍被空格路径卡住，可临时映射短路径：`subst X: "D:\programing Python\Python  document\CIFAR-10"`，用完 `subst X: /d` 解除。
- `numpy` 是 2.4.6。torch 2.5.1 对 numpy 2.x 兼容，自检已实际验证过，无需降级。
