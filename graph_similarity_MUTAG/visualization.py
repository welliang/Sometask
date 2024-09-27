import matplotlib.pyplot as plt
import networkx as nx

def visualize_supergraph(supergraph):
    # 设置图形大小
    plt.figure(figsize=(20, 20))

    # 获取节点度数
    node_degree = dict(supergraph.degree())

    # 设置节点大小和颜色
    node_size = [max(50, min(300, d * 2)) for d in node_degree.values()]
    node_color = list(node_degree.values())

    # 获取边权重
    edge_weights = [supergraph[u][v]['weight'] for u, v in supergraph.edges()]

    # 设置布局
    pos = nx.spring_layout(supergraph, k=0.9, iterations=50)

    # 绘制节点
    nx.draw_networkx_nodes(supergraph, pos, node_size=node_size, node_color=node_color, cmap=plt.cm.viridis, alpha=0.9)

    # 绘制边
    nx.draw_networkx_edges(supergraph, pos, width=edge_weights, alpha=0.5, edge_color='lightgray')

    # 绘制节点标签
    nx.draw_networkx_labels(supergraph, pos, font_size=6, font_color='white')

    # 设置颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_degree.values()), vmax=max(node_degree.values())))
    sm.set_array([])
    plt.colorbar(sm, label='Node Degree')

    # 设置标题和其他参数
    plt.title('Supergraph Visualization', fontsize=20)
    plt.axis('off')

    # 显示图形
    plt.tight_layout()
    plt.show()