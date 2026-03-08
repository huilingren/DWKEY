#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <set>
class Graph
{
public:
    Graph();
    Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to,const double* _in_nodes_weight,const double* _out_nodes_weight,const double* _betweenness,const double* _edge_weights);
    int num_nodes;
    int num_edges;
    std::vector< std::vector< int > > adj_list;
    std::vector< std::pair<int, int> > edge_list;
    std::vector<double> in_nodes_weight;
    std::vector<double> out_nodes_weight;
    std::vector<double> betweenness;
    double in_total_nodes_weight;
    double out_total_nodes_weight;
    double total_betweenness;
    std::vector<double> edge_weights;
    double edgetotalweight;
    double Iedgetotalweight;
    double Oedgetotalweight;
    std::vector<double> edge_weight_sum;
    std::vector<double> in_edge_weight_sum;
    std::vector<double> out_edge_sum;
    std::vector<double> in_edge_sum;
    double getTwoRankNeighborsRatio(std::vector<int> covered);
};

class GSet
{
public:
    GSet();
    void InsertGraph(int gid, std::shared_ptr<Graph> graph);
    std::shared_ptr<Graph> Sample();
    std::shared_ptr<Graph> Get(int gid);
    void Clear();
    std::map<int, std::shared_ptr<Graph> > graph_pool;
};

extern GSet GSetTrain;
extern GSet GSetTest;

#endif