#include "graph.h"
#include <cassert>
#include <iostream>
#include <random>
#include <iterator>
//#include "stdio.h"

Graph::Graph() : num_nodes(0), num_edges(0)
{
    edge_list.clear();
    adj_list.clear(); 
    in_nodes_weight.clear();
    out_nodes_weight.clear();
    betweenness.clear();
    in_total_nodes_weight= 0.0;
    out_total_nodes_weight= 0.0;
    total_betweenness = 0.0;
    edge_weights.clear();
    edgetotalweight = 0.0;
    edge_weight_sum.clear();
    in_edge_weight_sum.clear();
}

Graph::Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to,const double* _in_nodes_weight,const double* _out_nodes_weight,const double* _betweenness,const double* _edge_weights)
        : num_nodes(_num_nodes), num_edges(_num_edges)
{
    edge_list.resize(num_edges);
    adj_list.resize(num_nodes);
    in_nodes_weight.clear();
    out_nodes_weight.clear();
    betweenness.clear();
    in_total_nodes_weight = 0.0;
    out_total_nodes_weight = 0.0;
    total_betweenness = 0.0;
    edge_weights.clear();
    edgetotalweight = 0.0;
    edge_weight_sum.resize(num_nodes);
    in_edge_weight_sum.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
    {
        adj_list[i].clear();
        in_nodes_weight.push_back(_in_nodes_weight[i]);
        out_nodes_weight.push_back(_out_nodes_weight[i]);
        betweenness.push_back(_betweenness[i]);
        in_total_nodes_weight+=_in_nodes_weight[i];
        out_total_nodes_weight+=_out_nodes_weight[i];
        total_betweenness+=_betweenness[i];
    }

    for (int i = 0; i < num_edges; ++i)
    {
        int x = edges_from[i], y = edges_to[i];
        adj_list[x].push_back(y);
        edge_list[i] = std::make_pair(edges_from[i], edges_to[i]);
        edge_weights.push_back(_edge_weights[i]); 
        edgetotalweight += _edge_weights[i];  
        edge_weight_sum[x] += _edge_weights[i];  // x的出边权重和
        in_edge_weight_sum[y] += _edge_weights[i];  // y的入边权重和
        // out_edge_sum[x] += _edge_weights[i]; // 出边权重和
        // in_edge_sum[y] += _edge_weights[i];  // 入边权重和 
    }
}

double Graph::getTwoRankNeighborsRatio(std::vector<int> covered)
{
    std::set<int> tempSet;
    for(int i =0;i< (int)covered.size();++i){
        tempSet.insert(covered[i]);
    }
    double sum  = 0;
    for(int i =0;i<num_nodes;++i){
        if(tempSet.count(i)==0){
        for(int j=i+1;j<num_nodes;++j){
        if(tempSet.count(j)==0){
            std::vector<int> v3;
            std::set_intersection(adj_list[i].begin(),adj_list[i].end(),adj_list[j].begin(),adj_list[j].end(),std::inserter(v3,v3.begin()));
            if(v3.size()>0){
                sum += 1.0;
            }
        }
        }
        }
    }
    return sum;
}

GSet::GSet()
{
    graph_pool.clear();
}
void GSet::Clear()
{
    graph_pool.clear();
}

void GSet::InsertGraph(int gid, std::shared_ptr<Graph> graph)
{
    assert(graph_pool.count(gid) == 0);

    graph_pool[gid] = graph;
}

std::shared_ptr<Graph> GSet::Get(int gid)
{
    assert(graph_pool.count(gid));
    return graph_pool[gid];
}

std::shared_ptr<Graph> GSet::Sample()
{
//    printf("graph_pool_size:%d",graph_pool.size());
    assert(graph_pool.size());
//    printf("graph_pool_size:%d",graph_pool.size());
    int gid = rand() % graph_pool.size();
    assert(graph_pool[gid]);
    return graph_pool[gid];
}

GSet GSetTrain, GSetTest;