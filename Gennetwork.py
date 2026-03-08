import random
import networkx as nx
import numpy as np
import os
def gen_graph(n, m):
    g = nx.barabasi_albert_graph(n, m)
    g = nx.DiGraph(g)

    edge_remove = []
    for u, v in g.edges():
        if random.random() < 0.05:  
            edge_remove.append((u, v))
        elif random.random() > 0.95:
            edge_remove.append((v, u))
        
    g.remove_edges_from(edge_remove)
    # 添加边权重
    for u, v in g.edges():
        # 为边设置随机权重
        g[u][v]['weight'] = random.uniform(0.1, 1.0) 
        #degree = nx.degree_centrality(g) 
        #maxDegree = max(dict(degree).values())
        # use degree centrality
    in_degree = nx.in_degree_centrality(g)
    out_degree = nx.out_degree_centrality(g)
    nx.set_node_attributes(g, in_degree, 'in_degree_centrality')
    nx.set_node_attributes(g, out_degree, 'out_degree_centrality')
    return g

def save_graph_as_gml(g, i):
    # 删除复杂的节点属性，确保图的保存不包含不兼容的属性
    for node in g.nodes():
        if 'in_degree_centrality' in g.nodes[node]:
            del g.nodes[node]['in_degree_centrality']
        if 'out_degree_centrality' in g.nodes[node]:
            del g.nodes[node]['out_degree_centrality']

    # 确保保存目录存在
    if not os.path.exists('testdata'):
        os.makedirs('testdata')

    # 构造文件名
    file_name = f'testdata/g_{i}'

    # 将图保存为GML格式
    nx.write_gml(g, file_name)

if __name__ == "__main__":
   for i in range(0,100):
       n = random.randint(30,50)
       g = gen_graph(n, 4)
       save_graph_as_gml(g,i)