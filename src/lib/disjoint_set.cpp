#include "graph.h"
#include <cassert>
#include <iostream>
#include <random>
#include <iterator>
#include "disjoint_set.h"
#include <stack>
#include <fstream>  
//#include "stdio.h"

Disjoint_Set::Disjoint_Set(int graphSize)
{
    unionSet.resize(graphSize);
    rankCount.resize(graphSize);
    for (int i = 0; i < (int)unionSet.size(); i++)
    {
        unionSet[i] = i;
        rankCount[i] = 1;
    }
    maxRankCount = 1;
}

Disjoint_Set::~Disjoint_Set()
{
    unionSet.clear();
    rankCount.clear();
    maxRankCount = 1;
}

int Disjoint_Set::findRoot(int node)
{
    if (node != unionSet[node])
    {
        int rootNode = findRoot(unionSet[node]);
        unionSet[node] = rootNode;
        return rootNode;
    }
    else
    {
        return node;
    }
}

void Disjoint_Set::merge(int node1, int node2)
{
    int node1Root = findRoot(node1);
    int node2Root = findRoot(node2);
    if (node1Root != node2Root)
    {
        if (rankCount[node2Root] > rankCount[node1Root])
        {
            unionSet[node1Root] = node2Root;
            rankCount[node2Root] += rankCount[node1Root];

            if (rankCount[node2Root] > maxRankCount)
            {
                maxRankCount = rankCount[node2Root];
            }
        }
        else
        {
            unionSet[node2Root] = node1Root;
            rankCount[node1Root] += rankCount[node2Root];

            if (rankCount[node1Root] > maxRankCount)
            {
                maxRankCount = rankCount[node1Root];
            }

        }
    }
}

double Disjoint_Set::getBiggestComponentCurrentRatio() const
{
    return double(maxRankCount) / double(rankCount.size());
}

int Disjoint_Set::getRank(int rootNode) const
{
    return rankCount[rootNode];
}

// -----------------------------------------------------------这是以前的
// Disjoint_Set::Disjoint_Set(int graphSize) 
// {
//     unionSet.resize(graphSize);
//     rankCount.resize(graphSize);
//     visit.resize(graphSize, 0);
//     finish.resize(graphSize, 0);
//     edge.resize(graphSize);  // 初始化每个节点的边集合
//     redge.resize(graphSize); // 初始化每个节点的反向边集合
//     Set.resize(graphSize, 0);
//     k = 0;
//     cnt = 0;
//     maxRankCount = 1; // 当前最大连通分量大小，初始为1
//     DmaxRankCount = 1;
// }

// Disjoint_Set::~Disjoint_Set()
// {
//     unionSet.clear();
//     rankCount.clear();
//     maxRankCount = 1;
//     DmaxRankCount = 1;
// }
 
// void Disjoint_Set::addEdge(int node1, int node2) {
//         edge[node1].push_back(node2);
//         redge[node2].push_back(node1);
// }

//  // DFS，用于正向遍历图
// void Disjoint_Set::dfs(int v, const std::set<int>& covered_set)
// {
//     visit[v] = 1;  // v is visisted
//     for (auto u : edge[v]) { // 遍历节点v的邻居节点
//         if (visit[u] == 0 && covered_set.count(u) == 0) {  // 如果没有访问过，递归调用
//             dfs(u, covered_set);
//         }
//     }
//     finish[++cnt] = v; // 将v加入finished数组中
// }

// // 反向DFS，用于计算强连通分量
// void Disjoint_Set::rdfs(int v,int k, const std::set<int>& covered_set){
// 	visit[v] = 1; 
//     Set[v] = k;  // 将节点v标记为属于第k个强连通分量
//     // std::cout << "Node: " << v << " belongs to component: " << k << std::endl;
//     for (auto u : redge[v]) { 
//         if (visit[u] == 0 && covered_set.count(u) == 0) {
//             rdfs(u, k , covered_set); 
//         }
//     }
// } 

// void Disjoint_Set::Solve(int graphSize, const std::set<int>& covered_set){
// 	//  进行正向DFS
//     // 调整数组大小并初始化
//     visit.assign(graphSize, 0);
//     finish.assign(graphSize, -1); // 初始化为无效值
//     Set.resize(graphSize);
    
//     cnt = 0;  // 重置完成时间计数器

//     for (int i = 0; i < graphSize; i++) {
//         if (!visit[i] && !covered_set.count(i)) {
//             dfs(i, covered_set);
//         }
//     }

//     // 重置visit数组用于反向DFS
//     std::fill(visit.begin(), visit.end(), 0);
//     k = 0;

//     // 对节点进行反向DFS，计算强连通分量
//     // for (int i = graphSize-1; i >= 0; i--) {
//     //     int v = finish[i]; 
//     //     if (visit[v] == 0 && covered_set.count(v) == 0 ) {
//     //         k++;  // 新的一轮反向DFS开始，代表发现一个新的强连通分量
//     //         rdfs(v, k, covered_set);
//     //     }
//     // }
//     for (int i = cnt; i >= 1; --i) { // 修正循环范围
//         int v = finish[i];
//         if (!visit[v] && !covered_set.count(v)) {
//             ++k;
//             rdfs(v, k, covered_set);
//         }
//     }

//     std::vector<int> componentSize(k + 1, 0);  // 存储每个强连通分量的大小
//     for (int i = 0; i < graphSize; i++) {
//         if (!covered_set.count(i))
//         {
//             // std::cout << "Node " << i << " belongs to component " << Set[i] 
//             //             << ", component size: " << componentSize[Set[i]] << std::endl;
//             componentSize[Set[i]]++;  // 将节点i加入到它所属的强连通分量中
//         }
//     }
//     DmaxRankCount = *std::max_element(componentSize.begin(), componentSize.end());  // 获取最大强连通分量的大小
//     // std::cout << "Current Maximum Strongly Connected Component Size: " << DmaxRankCount << std::endl;
// }

// // void Disjoint_Set::Rsolve(int graphSize, int Rnode, std::vector<bool>& backupAllVex){  // 计算恢复时候的强连通分量
// // 	//  进行正向DFS
// //     int k = 0;
// //     int cnt = 0;
// //     Set.resize(graphSize, 0);
// //     finish.resize(graphSize, 0);
// //     std::fill(visit.begin(), visit.end(), 0);  // 重置visit数组
// //     for (int i = 0; i < graphSize; i++) {
// //         if (visit[i] == 0 && backupAllVex[i]) {
// //             dfs(i);
// //         }
// //     }
// //     // dfs(Rnode);  // 从恢复的节点开始DFS遍历

// //     // 重置visit数组用于反向DFS
// //     std::fill(visit.begin(), visit.end(), 0);

// //     // 对节点进行反向DFS，计算强连通分量
// //     for (int i = graphSize-1; i >= 0; i--) {
// //         int v = finish[i]; 
// //         if (visit[v] == 0 && backupAllVex[i]) {
// //             k++;  // 新的一轮反向DFS开始，代表发现一个新的强连通分量
// //             rdfs(v, k);
// //         }
// //     }

// //     std::vector<int> componentSize(k + 1, 0);  // 存储每个强连通分量的大小
// //     for (int i = 0; i < graphSize; i++) {
// //         if  (backupAllVex[i] && Set[i]!=0){
// //             componentSize[Set[i]]++; 
// //             //  std::cout << "Node " << i << " belongs to component " << Set[i] 
// //             //             << ", component size: " << componentSize[Set[i]] << std::endl;
// //         }
// //     }
// //     DmaxRankCount = *std::max_element(componentSize.begin(), componentSize.end());  // 获取最大强连通分量的大小
// // }


// int Disjoint_Set::findRoot(int node) // 找到node所在集合的根节点
// {
//     if (node != unionSet[node])  // 如果不是父节点，递归调用找到根节点
//     {
//         int rootNode = findRoot(unionSet[node]); // 找到根节点
//         unionSet[node] = rootNode; // 将当前节点直接连接到根节点
//         return rootNode;
//     }
//     else
//     {
//         return node; 
//     }
// }

// void Disjoint_Set::merge(int node1, int node2) // 合并两个节点所在集合
// { 
//     int node1Root = findRoot(node1);
//     int node2Root = findRoot(node2);
//     if (node1Root != node2Root)
//     {
//         double node1Rank = (double)rankCount[node1Root]; // 获取连通分量的大小
//         double node2Rank = (double)rankCount[node2Root];
//         // CCDScore = CCDScore - node1Rank*(node1Rank-1)/2.0 - node2Rank*(node2Rank-1)/2.0;   // 从当前 CCDScore 中减去两集合合并前的得分，因为原来的两个连通分量不存在。对于大小为 r 的连通分量，其得分为 r * (r - 1) / 2
//         // CCDScore = CCDScore + (node1Rank+node2Rank)*(node1Rank+node2Rank-1)/2.0;   // 将合并后的新连通分量的得分加回到 CCDScore
//         CCDScore = CCDScore - node1Rank*(node1Rank-1) - node2Rank*(node2Rank-1); // 不需要除以2,因为有向图的存在方向性，节点对组合为n*(n-1)
//         CCDScore = CCDScore + (node1Rank+node2Rank)*(node1Rank+node2Rank-1);
        
//         if (rankCount[node2Root] > rankCount[node1Root])  // 若 node2Root 所属连通分量的大小大于 node1Root，将 node1Root 合并到 node2Root
//         {
//             unionSet[node1Root] = node2Root;
//             rankCount[node2Root] += rankCount[node1Root];

//             if (rankCount[node2Root] > maxRankCount) // 若合并后的集合大小大于当前最大连通分量大小，更新 maxRankCount
//             {
//                 maxRankCount = rankCount[node2Root];
//             }
//         }
//         else
//         {
//             unionSet[node2Root] = node1Root;
//             rankCount[node1Root] += rankCount[node2Root];

//             if (rankCount[node1Root] > maxRankCount)  
//             {
//                 maxRankCount = rankCount[node1Root];
//             }
//         }
//     }
// }


// double Disjoint_Set::getBiggestComponentCurrentRatio() const  // 最大连通分量的大小除以总节点数
// {
//     return double(maxRankCount) / double(rankCount.size());
// }


// int Disjoint_Set::getRank(int rootNode) const // 获取某个根节点 rootNode 所属连通分量的大小
// {
//     return rankCount[rootNode];
// }

