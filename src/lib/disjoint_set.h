#ifndef DISJOINT_SET_H
#define DISJOINT_SET_H

#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <set>
class Disjoint_Set
{
public:
    Disjoint_Set();
    Disjoint_Set(int graphSize);
    ~Disjoint_Set();
    void reset(int graphSize);
    int findRoot(int node);
    void merge(int node1, int node2);
    double getBiggestComponentCurrentRatio() const;
    int getRank(int rootNode) const;
    void dfs(int v, const std::set<int>& covered_set);
    void rdfs(int v, int k, const std::set<int>& covered_set);
    void Solve(int graphSize, const std::set<int>& covered_set);
    // void Rsolve(int grapgSize, int Rnode, std::vector<bool>&);
    void addEdge(int node1, int node2);
	std::vector<int> unionSet; 
	std::vector<int> rankCount;
    std::vector<int> visit; // 记录节点是否被访问
    std::vector<int> finish; // 记录节点访问的结束顺序
    std::vector<std::vector<int>> edge;  // 正向边
    std::vector<std::vector<int>> redge; // 反向边
    std::vector<int> Set;  // 记录节点的强连通分量
	int maxRankCount;
    int DmaxRankCount;
    int cnt; // 访问节点的计数
    int k;    // 强连通分量的计数
	double CCDScore;
};



#endif