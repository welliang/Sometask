import torch
from data_loader import load_mutag_dataset
from gnn_embedding import GIN, get_graph_embeddings
from g2g_model import G2GSimilarityNetwork, train as train_g2g
from supergraph import build_supergraph
from msq_index import build_msq_index
from siamese_network import SiameseNetwork, train as train_siamese
from similarity_search import search
from evaluation import multi_dimension_evaluation
from visualization import visualize_supergraph

def main():
    # 1. 加载数据
    data_list = load_mutag_dataset()
    print("数据加载完成，共有 {} 个图".format(len(data_list)))

    # 2. 提取图嵌入
    gin_model = GIN(num_features=data_list[0].num_features)
    embeddings = get_graph_embeddings(gin_model, data_list)
    print("图嵌入提取完成")

    # 3. 训练 G2G 模型
    g2g_model = G2GSimilarityNetwork()
    train_g2g(g2g_model, embeddings)
    print("G2G 模型训练完成")

    # 4. 构建超图
    super_graph = build_supergraph(g2g_model, embeddings)
    print("超图构建完成，共有 {} 个节点和 {} 条边".format(super_graph.number_of_nodes(), super_graph.number_of_edges()))

    # 5. 构建 MSQ 索引
    graph_index = build_msq_index(super_graph, data_list)
    print("MSQ 索引构建完成")

    # 6. 训练 Siamese 网络
    siamese_model = SiameseNetwork()
    train_siamese(siamese_model, embeddings)
    print("Siamese 网络训练完成")

    # 7. 执行相似性搜索
    query_graph = 0  # 使用第一个图作为查询图
    similar_graphs = search(query_graph, graph_index, siamese_model, embeddings)
    print("相似性搜索完成，最相似的图：", similar_graphs[:5])

    # 8. 评估
    multi_dimension_evaluation(siamese_model, similar_graphs)

    # 9. 可视化
    visualize_supergraph(super_graph)

if __name__ == "__main__":
    main()