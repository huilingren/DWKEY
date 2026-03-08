import networkx as nx

def has_isolated_nodes(input_file):
    """
    检查图中是否存在孤立节点
    :param graph: NetworkX 图对象
    :return: 孤立节点的列表，如果没有孤立节点则返回空列表
    """
    # 使用 NetworkX 的 isolated_nodes 方法查找孤立节点
    graph = nx.read_gml(input_file)
    isolated_nodes = list(nx.isolates(graph))
    return isolated_nodes


# 检查是否存在孤立节点
input_file = "/home/renhuiling/Code_Ren/data/real/Cost/coin_degree.gml"
isolated = has_isolated_nodes(input_file)
if isolated:
    print(f"网络中存在孤立节点: {isolated}")
else:
    print("网络中没有孤立节点。")
