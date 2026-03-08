import networkx as nx

def reindex_gml(input_file, output_file):
    # 读取 GML 文件
    g = nx.read_gml(input_file)

    # 获取当前节点列表并排序
    original_nodes = sorted(g.nodes())
    print(f"Original nodes: {original_nodes}")

    # 创建新的节点编号映射，按照从 1 到 n 连续编号
    mapping = {original_node: new_id for new_id, original_node in enumerate(original_nodes, start=1)}

    print(f"Mapping: {mapping}")  # 输出映射关系

    # 重新编号图的节点
    g = nx.relabel_nodes(g, mapping)

    # 将新的图保存到输出文件
    nx.write_gml(g, output_file)
    print(f"Reindexed graph saved to {output_file}")

# 使用方法
input_file = "/home/renhuiling/Code_Ren/data/real/Cost/bitcoin_degree.gml"  # 输入的 GML 文件
output_file = "reindexed_output.gml"  # 输出的 GML 文件
reindex_gml(input_file, output_file)
