import networkx as nx
import torch

def build_supergraph(g2g_model, embeddings):
    supergraph = nx.Graph()
    num_graphs = len(embeddings)

    for i in range(num_graphs):
        supergraph.add_node(i, embedding=embeddings[i])

    for i in range(num_graphs):
        for j in range(i + 1, num_graphs):
            emb1 = torch.tensor(embeddings[i]).float()
            emb2 = torch.tensor(embeddings[j]).float()
            similarity = g2g_model(emb1, emb2).item()
            if similarity > 0.5:  # You can adjust this threshold
                supergraph.add_edge(i, j, weight=similarity)

    return supergraph

def get_node_features(supergraph, node):
    degree = supergraph.degree(node)
    clustering_coeff = nx.clustering(supergraph, node)
    degree_centrality = nx.degree_centrality(supergraph)[node]
    closeness_centrality = nx.closeness_centrality(supergraph)[node]
    betweenness_centrality = nx.betweenness_centrality(supergraph)[node]

    return {
        "degree": degree,
        "clustering_coeff": clustering_coeff,
        "degree_centrality": degree_centrality,
        "closeness_centrality": closeness_centrality,
        "betweenness_centrality": betweenness_centrality
    }