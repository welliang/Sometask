import numpy as np
from supergraph import get_node_features

def compute_q_gram(supergraph, node):
    q_gram = []
    node_feat = supergraph.nodes[node].get('embedding', None)
    if node_feat is not None:
        q_gram.append(node_feat)
    else:
        q_gram.append([0] * len(supergraph.nodes[0]['embedding']))

    for neighbor in supergraph.neighbors(node):
        neighbor_feat = supergraph.nodes[neighbor].get('embedding', None)
        if neighbor_feat is not None:
            q_gram.append(neighbor_feat)
        else:
            q_gram.append([0] * len(supergraph.nodes[0]['embedding']))

    return q_gram

def compute_msq_index_for_node(supergraph, node):
    q_gram = compute_q_gram(supergraph, node)
    msq_index = tuple(map(tuple, q_gram))
    return msq_index

def build_msq_index(supergraph, data_list):
    graph_index = {}
    for node in supergraph.nodes():
        msq_index = compute_msq_index_for_node(supergraph, node)
        supergraph_features = get_node_features(supergraph, node)
        graph_index[node] = {
            "msq_index": msq_index,
            "features": supergraph_features,
            "original_graph": data_list[node]
        }
    return graph_index