# Graph Similarity Search Project

该项目使用 GINConv 和 DGL 实现了一个图分类模型，兼容 Python 3.12。

## 环境要求

- Python 3.12
- PyTorch
- DGL
- NumPy
- scikit-learn
- tqdm

使用以下命令安装所需的 Python 包：

```bash
pip install -r requirements.txt
```
使用方法
训练模型：

```bash
python main.py --mode train
```

评估模型：

```bash
python main.py --mode evaluate
```
可以调整参数，例如训练轮数、批量大小和学习率：

```bash
python main.py --mode train --epochs 20 --batch_size 64 --learning_rate 0.0001
```
## 文件说明
- main.py：程序的主要入口，解析命令行参数并协调执行。
- data_processing.py：包含下载和准备数据集的函数。
- models.py：定义了图分类模型的结构。
- training.py：包含训练模型的函数。
- evaluation.py：提供评估模型的函数。
- utils.py：包含实用函数，例如设置随机种子。
- requirements.txt：列出了所需的 Python 包。
- 
## 数据集
代码将自动下载并使用 DGL 的 TUDataset 类提供的 TU Dortmund 大学的 AIDS 数据集。

**注意事项：**

- **数据集下载和读取：** 代码使用 DGL 的 `TUDataset` 类自动下载和读取 AIDS 数据集，无需手动下载。

- **Python 3.12 兼容性：** 所有代码均在 Python 3.12 环境下编写，并使用最新版本的相关库，确保兼容性。

- **库版本：** 请确保安装与 Python 3.12 兼容的 DGL 和 PyTorch 版本。例如：

  ```bash
  pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cpu
  pip install dgl==1.0.1
