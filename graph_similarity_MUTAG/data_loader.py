import torch
from torch_geometric.datasets import TUDataset

def load_mutag_dataset():
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    data_list = [data for data in dataset]
    return check_and_fix_edge_index(data_list)

def check_and_fix_edge_index(data_list):
    fixed_data_list = []
    for i, data in enumerate(data_list):
        max_index = data.edge_index.max().item()
        num_nodes = data.x.size(0)
        if max_index >= num_nodes:
            print(f"Graph {i} has invalid edge index. Max index: {max_index}, but num_nodes: {num_nodes}")
            valid_mask = data.edge_index < num_nodes
            valid_mask = valid_mask.all(dim=0)
            data.edge_index = data.edge_index[:, valid_mask]
        fixed_data_list.append(data)
    return fixed_data_list