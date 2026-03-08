#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <set>
#include "disjoint_set.h"
#include <stack>


class GraphUtil
{
public:
    GraphUtil();

    ~GraphUtil();

    void deleteNode(std::vector<std::vector<int> > &adjListGraph, int node);

     void recoverAddNode(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex, std::vector<std::vector<int> > &adjListGraph, int node, Disjoint_Set &unionSet);

     void addEdge(std::vector<std::vector<int> > &adjListGraph, int node0, int node1);

     void addEdge1(std::vector<std::vector<int> > &adjListGraph, int node0, int node1);
    // 检查是否存在反向边
    bool hasReverseEdge(const std::vector<std::vector<int>>& graph, int src, int dest);

    // 检查两节点是否通过路径互相可达
    bool isMutuallyReachable(const std::vector<std::vector<int>>& graph, int node1, int node2);

    // 使用 DFS 检查从 source 是否能到达 target
    bool dfsReachable(const std::vector<std::vector<int>>& graph, int source, int target);


};



#endif