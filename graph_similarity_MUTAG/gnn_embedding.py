import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool

class GIN(torch.nn.Module):
    def __init__(self, num_features):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(num_features, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64)
            )
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64)
            )
        )
        self.lin = nn.Linear(64, 64)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return x

def get_graph_embeddings(model, data_list):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in data_list:
            data = data.to('cpu')
            x, edge_index, batch = data.x, data.edge_index, data.batch
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long)
            emb = model(x, edge_index, batch)
            embeddings.append(emb.numpy())
    return embeddings