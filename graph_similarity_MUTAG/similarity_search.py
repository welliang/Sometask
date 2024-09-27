import torch
import numpy as np


def compute_msq_similarity(msq_index1, msq_index2):
    # 将 numpy 数组转换为元组
    set1 = set(tuple(map(tuple, arr)) for arr in msq_index1)
    set2 = set(tuple(map(tuple, arr)) for arr in msq_index2)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def compute_feature_similarity(query_features, target_features):
    similarity = 0.0
    weights = {
        "degree": 0.2,
        "clustering_coeff": 0.2,
        "degree_centrality": 0.2,
        "closeness_centrality": 0.2,
        "betweenness_centrality": 0.1
    }
    for key in weights:
        similarity += weights[key] * (1 - abs(query_features[key] - target_features[key]))
    return similarity


def search(query_graph, graph_index, siamese_model, embeddings, top_k=5):
    query_embedding = torch.tensor(embeddings[query_graph]).float()
    query_index_data = graph_index[query_graph]

    initial_similarities = {}
    for node, index_data in graph_index.items():
        if node == query_graph:
            continue
        msq_similarity = compute_msq_similarity(query_index_data["msq_index"], index_data["msq_index"])
        feature_similarity = compute_feature_similarity(query_index_data["features"], index_data["features"])
        total_similarity = msq_similarity + feature_similarity
        initial_similarities[node] = total_similarity

    sorted_initial = sorted(initial_similarities.items(), key=lambda item: item[1], reverse=True)[:top_k]
    candidate_indices = [node for node, _ in sorted_initial]

    final_similarities = []
    siamese_model.eval()
    with torch.no_grad():
        for idx in candidate_indices:
            candidate_embedding = torch.tensor(embeddings[idx]).float()
            similarity = siamese_model(query_embedding.unsqueeze(0), candidate_embedding.unsqueeze(0)).item()
            final_similarities.append((idx, similarity))

    return sorted(final_similarities, key=lambda x: x[1], reverse=True)