#include <iostream>
#include "disjoint_set.h"
#include <set>
#include <vector>
#include <random>
#include <ctime>

int main() {
  // 初始化图，包含 5 个节点
    int graphSize = 6;
    Disjoint_Set ds(graphSize);
    // 添加双向边
    ds.addEdge(0, 1); // 添加 0 -> 1
    ds.addEdge(1, 0); // 添加 1 -> 0

    ds.addEdge(1, 2); // 添加 1 -> 2
    ds.addEdge(2, 1); // 添加 2 -> 1

    ds.addEdge(3, 4); // 添加 3 -> 4
    ds.addEdge(4, 3); // 添加 4 -> 3

    // 添加单向边
    ds.addEdge(2, 3); // 添加 2 -> 3

    ds.addEdge(5,0);
    ds.addEdge(0,5);

   // 初始化所有节点
    std::vector<int> nodes(graphSize);
    for (int i = 0; i < graphSize; ++i) {
        nodes[i] = i; // 节点编号为 0 到 graphSize-1
    }
    // 随机打乱节点顺序
    std::srand(std::time(nullptr)); // 设置随机种子
    std::random_shuffle(nodes.begin(), nodes.end());

    // 定义 covered_set 为 0-4 的节点
    std::set<int> covered_set;

    for (int i = 0; i < graphSize; ++i){
        Disjoint_Set temp_ds(graphSize);
        covered_set.insert(nodes[i]);
        std::cout<< "node[i] is" << nodes[i] <<std::endl;
        for (int node = 0; node < graphSize; node++) // 计算除了覆盖节点之外的图的连通性
            {
                if (covered_set.count(node) == 0)  // 选择的第一个节点此时已经不会在计算连通性了，因为此时不等于0，就能计算去除此节点的奖励
                {
                    for (auto neigh : ds.edge[node])  
                    {
                        std::cout<<"node " << node <<" neigh is" << neigh << std::endl;
                        if (covered_set.count(neigh) == 0) 
                        {
                            // 添加边 分别添加正向边和反向边，用于dfs和rdfs，此时已经是去除覆盖节点之后的图了
                            temp_ds.addEdge(node, neigh);
                        }
                    }
                }
            }
        // 构造 covered_set，仅包含当前被删除的节点
        

        // 调用 Solve 函数计算强连通分量
        temp_ds.Solve(graphSize, covered_set);

        // 输出去除节点后的结果
        std::cout << "After removing node " << i << ":" << std::endl;

        // 输出每个节点所属的强连通分量
        for (int j = 0; j < graphSize; ++j) {
            if (covered_set.count(j) == 0) { // 只打印未被删除的节点
                std::cout << "Node " << j << " belongs to component " << temp_ds.Set[j] << std::endl;
            }
        }

        // 输出当前的最大强连通分量大小
        std::cout << "Maximum Strongly Connected Component Size: "
                  << temp_ds.DmaxRankCount << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
    }

    return 0;

    }