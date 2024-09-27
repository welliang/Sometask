# MUTAG 数据集上的图相似度搜索

本项目实现了一个基于 MUTAG 数据集的图相似度搜索系统。它结合了图神经网络（GNN）、图到图（G2G）相似度网络和孪生网络等多种技术，以实现高效和准确的图相似度搜索。

## 项目结构

本项目包含以下主要组件：

1. `data_loader.py`: 加载和预处理 MUTAG 数据集。
2. `gnn_embedding.py`: 实现图同构网络（GIN）用于图嵌入。
3. `g2g_model.py`: 实现并训练图到图相似度网络。
4. `supergraph.py`: 基于图相似度构建超图。
5. `msq_index.py`: 构建 MSQ（多尺度二次）索引以实现高效相似度搜索。
6. `siamese_network.py`: 实现并训练孪生网络，用于精细化相似度计算。
7. `similarity_search.py`: 使用训练好的模型和索引执行实际的相似度搜索。
8. `evaluation.py`: 提供搜索结果的多维度评估函数。
9. `visualization.py`: 提供超图的可视化功能。
10. `main.py`: 协调整个工作流程。

## 安装

1. 克隆此仓库：
   ```
   git clone <仓库URL>
   cd <仓库名称>
   ```

2. 安装所需依赖：
   ```
   pip install torch torch_geometric networkx matplotlib numpy scipy scikit-learn
   ```

## 使用方法

运行主脚本以执行整个工作流程：

```
python main.py
```

这将加载 MUTAG 数据集，训练模型，构建超图和 MSQ 索引，执行相似度搜索，并评估结果。

## 依赖项

- PyTorch
- PyTorch Geometric
- NetworkX
- Matplotlib
- NumPy
- SciPy
- scikit-learn

## 数据集

本项目使用 MUTAG 数据集，这是一个由致突变性芳香族和杂环芳香族硝基化合物组成的集合。运行脚本时会自动下载该数据集。

## 自定义

您可以在 `main.py` 文件中修改各种参数来自定义系统行为，如训练轮数、学习率或要检索的相似图的数量。

## 评估

系统执行多维度评估，考虑搜索时间、内存使用和各种准确度指标等因素。执行过程中会将结果打印到控制台。

## 可视化

本项目包含构建超图的可视化功能，可以提供数据集整体结构和图之间关系的洞察。


## 联系方式

[微信：Lv215w]

---

关于每个组件的更详细信息，请参阅项目中的各个 Python 文件。
