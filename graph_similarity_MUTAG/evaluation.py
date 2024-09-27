import time
import tracemalloc
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import precision_score, recall_score, f1_score
import torch

def multi_dimension_evaluation(siamese_model, similar_graphs):
    # 1. 评估时间复杂度
    start_time = time.time()
    # 假设这里执行了一些操作
    end_time = time.time()
    print(f"执行时间: {end_time - start_time:.4f}秒")

    # 2. 评估空间复杂度
    tracemalloc.start()
    # 假设这里执行了一些操作
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"内存使用峰值: {peak / 10**6:.2f} MB")

    # 3. 精度指标评估
    # 假设我们有真实的相似度标签
    true_labels = np.random.rand(len(similar_graphs))
    predicted_labels = np.array([sim for _, sim in similar_graphs])

    # Spearman和Kendall相关系数
    spearman_corr, _ = spearmanr(true_labels, predicted_labels)
    kendall_corr, _ = kendalltau(true_labels, predicted_labels)
    print(f"Spearman 相关系数: {spearman_corr:.4f}")
    print(f"Kendall 相关系数: {kendall_corr:.4f}")

    # 将连续的相似度转换为二元标签
    threshold = 0.5
    true_binary = (true_labels >= threshold).astype(int)
    pred_binary = (predicted_labels >= threshold).astype(int)

    # 计算精确度、召回率和F1分数
    precision = precision_score(true_binary, pred_binary)
    recall = recall_score(true_binary, pred_binary)
    f1 = f1_score(true_binary, pred_binary)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")