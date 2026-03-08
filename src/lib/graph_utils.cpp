#include "graph_utils.h"
#include <cassert>
#include <iostream>
#include <random>
#include <iterator>
//#include "stdio.h"

GraphUtil::GraphUtil()
{

}

GraphUtil::~GraphUtil()
{

}

 void GraphUtil::deleteNode(std::vector<std::vector<int> > &adjListGraph, int node)
{
//    for (auto neighbour : adjListGraph[node])
//    {
    for (int i = 0;i<(int)adjListGraph[node].size();++i)
    {
        int neighbour =  adjListGraph[node][i];
        adjListGraph[neighbour].erase(remove(adjListGraph[neighbour].begin(), adjListGraph[neighbour].end(), node), adjListGraph[neighbour].end());
    }

    adjListGraph[node].clear();
}

// void GraphUtil::recoverAddNode(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex, std::vector<std::vector<int> > &adjListGraph, int node, Disjoint_Set &unionSet)
// {

//     for (int i = 0; i < (int)backupCompletedAdjListGraph[node].size(); i++)
//     {
//         int neighbourNode = backupCompletedAdjListGraph[node][i];

//         if (backupAllVex[neighbourNode])  // 如果neighbour已经被恢复了才恢复边
//         {
//             addEdge(adjListGraph, node, neighbourNode);
//             unionSet.merge(node, neighbourNode);
//         }
//     }

//     backupAllVex[node] = true;
// }

// ----------------------------------------------------------------------------------------------------
// void GraphUtil::recoverAddNode(const std::vector<std::vector<int>>& backupCompletedAdjListGraph, 
//                                std::vector<bool>& backupAllVex, 
//                                std::vector<std::vector<int>>& adjListGraph, 
//                                int node, 
//                                Disjoint_Set& unionSet
//                                ) {
//     for (int i = 0; i < (int)backupCompletedAdjListGraph[node].size(); i++) {
//         int neighbourNode = backupCompletedAdjListGraph[node][i];

//         // 1. 如果邻居节点已恢复
//         if (backupAllVex[neighbourNode]) {
//             // 添加边 A -> B
//             addEdge(adjListGraph, node, neighbourNode);
//             // printf("88888");
//             // 如果反向边 B -> A 未恢复，则补充
//             if (!hasReverseEdge(adjListGraph, neighbourNode, node)) {
//                 addEdge(adjListGraph, neighbourNode, node);  // 添加反向边 B -> A
//             }
//             // add edge to compute LGCC
//             unionSet.addEdge(node, neighbourNode);
//         }
//     }

//     // 标记当前节点为已恢复
//     backupAllVex[node] = true;
// }

// // 检查是否存在反向边
// bool GraphUtil::hasReverseEdge(const std::vector<std::vector<int>>& graph, int src, int dest) {
//     for (int neighbor : graph[src]) {
//         if (neighbor == dest) {
//             return true;  // 找到反向边
//         }
//     }
//     return false;
// }

// // 检查两节点是否通过路径互相可达
// bool GraphUtil::isMutuallyReachable(const std::vector<std::vector<int>>& graph, int node1, int node2) {
//     return dfsReachable(graph, node1, node2) && dfsReachable(graph, node2, node1);
// }

// // 使用 DFS 检查从 source 是否能到达 target
// bool GraphUtil::dfsReachable(const std::vector<std::vector<int>>& graph, int source, int target) {
//     std::vector<bool> visited(graph.size(), false);
//     std::stack<int> stack;
//     stack.push(source);

//     while (!stack.empty()) {
//         int current = stack.top(); 
//         stack.pop();

//         if (current == target) {
//             return true;
//         }

//         if (!visited[current]) { 
//             visited[current] = true; 
//             for (int neighbor : graph[current]) {
//                 if (!visited[neighbor]) { 
//                     stack.push(neighbor);
//                 }
//             }
//         }
//     }

//     return false;
// }
// -----------------------------------------------------------------------------------------

void GraphUtil::recoverAddNode(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex, std::vector<std::vector<int> > &adjListGraph, int node, Disjoint_Set &unionSet)
{

    for (int i = 0; i < (int)backupCompletedAdjListGraph[node].size(); i++)
    {
        int neighbourNode = backupCompletedAdjListGraph[node][i];

        if (backupAllVex[neighbourNode])
        {
            addEdge1(adjListGraph, node, neighbourNode);
            unionSet.merge(node, neighbourNode);
        }
    }

    backupAllVex[node] = true;
}

void GraphUtil::addEdge1(std::vector<std::vector<int> > &adjListGraph, int node0, int node1) // 为了和prepare进行区分，只在计算鲁棒性的时候算作无向图
{
    if (((int)adjListGraph.size() - 1) < std::max(node0, node1))
    {
        adjListGraph.resize(std::max(node0, node1) + 1);
    }

    adjListGraph[node0].push_back(node1);  
    adjListGraph[node1].push_back(node0);
}


void GraphUtil::addEdge(std::vector<std::vector<int> > &adjListGraph, int node0, int node1)
{
    if (((int)adjListGraph.size() - 1) < std::max(node0, node1))
    {
        adjListGraph.resize(std::max(node0, node1) + 1);
    }

    adjListGraph[node0].push_back(node1);  
  //  adjListGraph[node1].push_back(node0);
}

