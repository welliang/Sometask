# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GINConv

class GraphClassificationModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=128):
        super(GraphClassificationModel, self).__init__()
        self.embedding_dim = embedding_dim
        # 定义 GINConv 层
        self.conv1 = GINConv(nn.Linear(1, embedding_dim), aggregator_type='sum')
        self.conv2 = GINConv(nn.Linear(embedding_dim, embedding_dim), aggregator_type='sum')
        # 定义全连接层
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)

    def forward(self, g):
        # 节点特征存储在 g.ndata['feat'] 中
        h = g.ndata['feat'].float()
        h = self.conv1(g, h)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # 图的表示
        out = self.fc1(hg)
        out = F.relu(out)
        out = self.fc2(out)
        return out  # 返回 logits
