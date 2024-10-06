# data_processing.py

import torch
import dgl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dgl.data import TUDataset

def download_and_prepare_data(dataset_name):
    if dataset_name.lower() == 'aids':
        dataset = download_aids_dataset()
        return dataset
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

def download_aids_dataset():
    # 使用 DGL 的 TUDataset 来加载 AIDS 数据集
    dataset = TUDataset(name='AIDS')
    return dataset

def prepare_dataloader(dataset, mode='train', batch_size=32):
    # 划分数据集为训练集和测试集
    num_samples = len(dataset)
    train_ratio = 0.8
    num_train = int(num_samples * train_ratio)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    if mode == 'train':
        selected_indices = indices[:num_train]
    else:
        selected_indices = indices[num_train:]
    subset = [dataset[i] for i in selected_indices]
    # 确保节点特征存在
    for i in range(len(subset)):
        g, label = subset[i]
        if 'feat' not in g.ndata:
            # 为节点赋予常数特征
            num_nodes = g.number_of_nodes()
            g.ndata['feat'] = torch.ones(num_nodes, 1)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=(mode=='train'), collate_fn=collate)
    return dataloader

def collate(samples):
    # 输入 `samples` 是一个列表，包含 (graph, label)
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels
