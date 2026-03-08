from mvc_env import MvcEnv

graph = MvcEnv.Graph()  # 假设 Graph 已暴露
graph.num_nodes = 5
graph.adj_list = [[1, 2], [0, 3], [0], [1], []]  # 自定义邻接表

env = MvcEnv(graph)
result = env.getMaxConnectedNodesNum()
print("最大连通分量节点数量:", result)
