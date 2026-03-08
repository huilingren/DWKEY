#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <set>
#include <memory>
#include "graph.h"
#include "disjoint_set.h"
#include "graph_utils.h"
#include "decrease_strategy.cpp"

class Utils
{
public:
    Utils();

    double getRobustness2(std::shared_ptr<Graph> graph,std::vector<int>);
    
    double getRobustness(std::shared_ptr<Graph> graph,std::vector<int>);

    double getRobustness1(std::shared_ptr<Graph> graph,std::vector<int>);

    std::vector<int> reInsert(std::shared_ptr<Graph> graph,std::vector<int> solution,const std::vector<int> allVex,int decreaseStrategyID,int reinsertEachStep);

    std::vector<int> reInsert_inner(const std::vector<int> &beforeOutput, std::shared_ptr<Graph> &graph, const std::vector<int> &allVex, std::shared_ptr<decreaseComponentStrategy> &decreaseStrategy,int reinsertEachStep);

    int getMxWccSz(std::shared_ptr<Graph> graph);

    std::vector<double> Betweenness(std::shared_ptr<Graph> _g);

    std::vector<double> Betweenness(std::vector< std::vector <int> > adj_list);

    std::vector<double> MaxWccSzList;

    std::set<int> covered_set;

    std::vector<double> CostRatios;

    std::vector<double> CostRatios1;
};

#endif