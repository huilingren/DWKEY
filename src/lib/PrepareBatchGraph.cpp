#include "PrepareBatchGraph.h"

sparseMatrix::sparseMatrix()
{
    rowNum = 0;
    colNum = 0;
}

sparseMatrix::~sparseMatrix()
{
    rowNum = 0;
    colNum = 0;
    rowIndex.clear();
    colIndex.clear();
    value.clear();
}

 PrepareBatchGraph::PrepareBatchGraph(int _aggregatorID)
{
    aggregatorID = _aggregatorID;
}

PrepareBatchGraph::~PrepareBatchGraph()
{
    act_select =nullptr;
    rep_global =nullptr;
    n2nsum_param =nullptr;
    subgsum_param =nullptr;
    laplacian_param = nullptr;
    idx_map_list.clear();
    node_feat.clear();
    aux_feat.clear();
    avail_act_cnt.clear();
    aggregatorID = -1;
}

int PrepareBatchGraph::GetStatusInfo(std::shared_ptr<Graph> g, int num, const int* covered,int& counter,int& twohop_number,int& threehop_number, std::vector<int>& idx_map)
{
    std::set<int> c;

    idx_map.resize(g->num_nodes);

    for (int i = 0; i < (int)g->num_nodes; ++i)
        idx_map[i] = -1; // 初始化为-1 表明所有节点没有被覆盖

    for (int i = 0; i < num; ++i)
        c.insert(covered[i]);  // 将covered所有节点添加到c集合中

    counter = 0;

    twohop_number = 0;
    threehop_number = 0;
    std::set<int> node_twohop_set;

    int n = 0;
 	std::map<int,int> node_twohop_counter;

    for (auto& p : g->edge_list) // 循环遍历每一条边p
    {
        if (c.count(p.first) || c.count(p.second)) // 对于一条边，如果两个节点中的任何一个在c集合，那么counter递增
        {
            counter++;  // 跳过被covered覆盖的节点        
        } else {
            if (idx_map[p.first] < 0)
                n++; // 统计没有被覆盖的节点

            if (idx_map[p.second] < 0)
                n++;
            // 如果p.first或p.second没有被covered覆盖，则被设置为0 如果都为-1表示所有的节点都被覆盖了
            // std::cout << " here idx_map is working " << std::endl;
            idx_map[p.first] = 0;
            idx_map[p.second] = 0;
            if (node_twohop_counter.find(p.first) != node_twohop_counter.end()) // 如果p.first在两跳里面，也就是已经有边指向p.first
            {
                twohop_number += node_twohop_counter[p.first]; 
            }
            node_twohop_counter[p.second] += 1; // 更新 p.second 的度
        }
    }   
    // std::cout<< " n is : " << n <<std::endl;
    return n; // 返回当前未被覆盖的节点数量
}

int PrepareBatchGraph::getMaxConnectedNodesNum(std::shared_ptr<Graph> g,
                                               const std::vector<int>& covered_vec
                                            )
{
    assert(g);
    std::unordered_set<int> covered(covered_vec.begin(), covered_vec.end());
    Disjoint_Set disjoint_Set =  Disjoint_Set(g->num_nodes);

    for (int i = 0; i < g->num_nodes; i++)
    {
        if (covered.count(i) == 0)
        {
            for (auto neigh : g->adj_list[i])
            {
                if (covered.count(neigh) == 0)
                {
                    disjoint_Set.merge(i, neigh);
                }
            }
        }
    }
    return disjoint_Set.maxRankCount;
}

void PrepareBatchGraph::SetupGraphInput(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list, 
                           std::vector< std::vector<int> > covered, 
                           const int* actions)
{
    act_select = std::shared_ptr<sparseMatrix>(new sparseMatrix()); // 存储动作选择的稀疏矩阵
    rep_global = std::shared_ptr<sparseMatrix>(new sparseMatrix()); // 存储全局表示的稀疏矩阵

    idx_map_list.resize(idxes.size());
    avail_act_cnt.resize(idxes.size()); // 存储每个图的可用动作数量


    int node_cnt = 0; // 节点数量

    for (size_t i = 0; i < (int)idxes.size(); ++i) // 遍历每个图索引
    {   
        // std::cout << "idxes[" << i << "] = " << idxes[i] << std::endl;
        std::vector<double> temp_feat;
        
        auto g = g_list[idxes[i]];
        int counter;
        int twohop_number;
        int threehop_number;

        temp_feat.push_back((double)getMaxConnectedNodesNum(g, covered[idxes[i]]) / (double)g->num_nodes); 
        // std::cout << "retio is" << (double)getMaxConnectedNodesNum(g, covered[idxes[i]]) / (double)g->num_nodes << std::endl;
        if (g->num_nodes)

            temp_feat.push_back((double)covered[idxes[i]].size() / (double)g->num_nodes);
            // 计算当前图g的可用动作数量，填充并返回
            avail_act_cnt[i] = GetStatusInfo(g, covered[idxes[i]].size(), covered[idxes[i]].data(), counter,twohop_number,threehop_number, idx_map_list[i]);

        if (g->edge_list.size())

            temp_feat.push_back((double)counter / (double)g->edge_list.size());

         temp_feat.push_back((double)twohop_number / ((double)g->num_nodes * (double)g->num_nodes));

         temp_feat.push_back(1.0);

        node_cnt += avail_act_cnt[i]; // 可选动作数量（节点数）
        aux_feat.push_back(temp_feat);
    }

    graph.Resize(idxes.size(), node_cnt);
    if (actions)
    {
        act_select->rowNum=idxes.size(); // 图的数量
        act_select->colNum=node_cnt;
    } else
    {
        rep_global->rowNum=node_cnt;
        rep_global->colNum=idxes.size();
    }

    node_cnt = 0;
    int edge_cnt = 0;
    // std::cout<< "node_feat:" <<node_feat.size() << " idxes "<< idxes.size()<<std::endl;
    // 添加节点和边
    for (size_t i = 0; i < (int)idxes.size(); ++i)
    {             
        auto g = g_list[idxes[i]];
        auto idx_map = idx_map_list[i];

        int t = 0;

        // std::cout<< "g->num_nodes:" <<g->num_nodes<<std::endl;
        for (int j = 0; j < g->num_nodes; ++j)
        {   
            if (idx_map[j] < 0)
                continue; // 如果idx_map都为-1，则加入不了节点
            idx_map[j] = t;// 重新映射节点索引
            // std::cout << "g->in_nodes_weight[j]  is" << g->in_nodes_weight[j] << std::endl;
            std::vector<double> temp_node_feat;
            // temp_node_feat.push_back(g->in_nodes_weight[j]);
            // temp_node_feat.push_back(g->out_nodes_weight[j]);
            temp_node_feat.push_back(g->betweenness[j]);
            // temp_node_feat.push_back(g->in_edge_weight_sum[j] + g->in_edge_sum[j]);
            temp_node_feat.push_back(1.0);
            node_feat.push_back(temp_node_feat);

            graph.AddNode(i, node_cnt + t); // 添加节点
            if (!actions)
            {
                rep_global->rowIndex.push_back(node_cnt + t);
                rep_global->colIndex.push_back(i);
                rep_global->value.push_back(1.0);
            }
            t += 1;
        }
        assert(t == avail_act_cnt[i]);
        
        if (actions)
        {   
            auto act = actions[idxes[i]];
            assert(idx_map[act] >= 0 && act >= 0 && act < g->num_nodes);
            act_select->rowIndex.push_back(i);
            act_select->colIndex.push_back(node_cnt + idx_map[act]);
            act_select->value.push_back(1.0);
        }
        
        for (size_t e_idx = 0; e_idx < g->edge_list.size(); ++e_idx)
        {   
            auto p = g->edge_list[e_idx];
            if (idx_map[p.first] < 0 || idx_map[p.second] < 0)
                continue;
            auto x = idx_map[p.first] + node_cnt, y = idx_map[p.second] + node_cnt;
            double weight = g->edge_weights[e_idx];
            graph.AddEdge(edge_cnt, x, y, weight);
            edge_cnt += 1;
        }
        node_cnt += avail_act_cnt[i];
        // std::cout << "Total nodes in graph: " << graph.num_nodes << std::endl;
        // std::cout << "Total edges in graph: " << graph.num_edges << std::endl;
    }

    assert(node_cnt == (int)graph.num_nodes);

    auto result_list = n2n_construct(&graph,aggregatorID);
    n2nsum_param = result_list[0];
    laplacian_param = result_list[1];
    subgsum_param = subg_construct(&graph,subgraph_id_span);
}

void PrepareBatchGraph::SetupTrain(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list,
                           std::vector< std::vector<int> > covered,
                           const int* actions)
{
    SetupGraphInput(idxes, g_list, covered, actions);
}



void PrepareBatchGraph::SetupPredAll(std::vector<int> idxes,
                           std::vector< std::shared_ptr<Graph> > g_list,
                           std::vector< std::vector<int> > covered)
{
    SetupGraphInput(idxes, g_list, covered, nullptr);
}

std::vector<std::shared_ptr<sparseMatrix>> n2n_construct(GraphStruct* graph,int aggregatorID)
{
    //aggregatorID = 0 sum
    //aggregatorID = 1 mean
    //aggregatorID = 2 GCN
    std::vector<std::shared_ptr<sparseMatrix>> resultList;
    resultList.resize(2);
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_nodes;
    result->colNum = graph->num_nodes;

    std::shared_ptr<sparseMatrix> result_laplacian= std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result_laplacian->rowNum = graph->num_nodes;
    result_laplacian->colNum = graph->num_nodes;

    std::vector<double> in_sum(graph->num_nodes, 0.0);

	for (unsigned int i = 0; i < graph->num_nodes; ++i)
	{
		auto& list = graph->in_edges->head[i];
        double diag_degree = 0.0;
        double sum = 0.0;
        for (const auto& edge : list) {
            int e_idx = edge.second;  // 获取该边在 edge_list 中的索引
            in_sum[i] += graph->Edge_weights[e_idx];
        }
        if (sum != 0.0 || !list.empty()) { 
            // result_laplacian->value.push_back(diag_degree);
            result_laplacian->value.push_back(sum);
            result_laplacian->rowIndex.push_back(i);
            result_laplacian->colIndex.push_back(i);
        }

		for (size_t j = 0; j < list.size(); ++j)
		{
            int e_idx = list[j].second;
            double edge_weight = graph->Edge_weights[e_idx];
            double val = 0.0;
		    switch(aggregatorID){
		       case 0:
		       {
		        //   val = edge_weight;
                  val = (in_sum[i] > 0) ? (edge_weight / in_sum[i]) : 0.0;
                //   std::cout << "val is" << val << std::endl;
		          break;
		       }
		       case 1:
		       {
		          result->value.push_back(1.0/(double)list.size());
		          break;
		       }
		       default:
		          break;
		    }
            result->value.push_back(val);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].second);

            result_laplacian->value.push_back(-val);
		    result_laplacian->rowIndex.push_back(i);
		    result_laplacian->colIndex.push_back(list[j].second);

		}
	}
	resultList[0]= result;
	resultList[1] = result_laplacian;
    return resultList;
}


std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_nodes;
    result->colNum = graph->num_edges;
	for (unsigned int i = 0; i < graph->num_nodes; ++i)
	{
		auto& list = graph->in_edges->head[i];
		for (size_t j = 0; j < (int)list.size(); ++j)
		{
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].first);
		}
	}
    return result;
}

std::shared_ptr<sparseMatrix> n2e_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_edges;
    result->colNum = graph->num_nodes;

	for (unsigned int i = 0; i < graph->num_edges; ++i)
	{
        result->value.push_back(1.0);
        result->rowIndex.push_back(i);
        result->colIndex.push_back(graph->edge_list[i].first);
	}
    return result;
}

std::shared_ptr<sparseMatrix> e2e_construct(GraphStruct* graph)
{
    std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_edges;
    result->colNum = graph->num_edges;
    for (unsigned int i = 0; i < graph->num_edges; ++i)
    {
        int node_from = graph->edge_list[i].first, node_to = graph->edge_list[i].second;
        auto& list = graph->in_edges->head[node_from];
        for (size_t j = 0; j < (int)list.size(); ++j)
        {
            if (list[j].second == node_to)
                continue;
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j].first);
        }
    }
    return result;
}

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph, std::vector<std::pair<int,int>>& subgraph_id_span)
{
   std::shared_ptr<sparseMatrix> result =std::shared_ptr<sparseMatrix>(new sparseMatrix());
    result->rowNum = graph->num_subgraph;
    result->colNum = graph->num_nodes;

    subgraph_id_span.clear();
    int start = 0;
    int end = 0;
	for (unsigned int i = 0; i < (int)graph->num_subgraph; ++i)
	{
		auto& list = graph->subgraph->head[i];
        end  = start + list.size() - 1;
		for (size_t j = 0; j < (int)list.size(); ++j)
		{
            double edge_weight = graph->Edge_weights[j];
            // std::cout<< "edge_weight " <<edge_weight <<std::endl;
            result->value.push_back(1.0);
            result->rowIndex.push_back(i);
            result->colIndex.push_back(list[j]);
		}
		if(list.size()>0){
		    subgraph_id_span.push_back(std::make_pair(start,end));
		}
		else{
		    subgraph_id_span.push_back(std::make_pair(graph->num_nodes,graph->num_nodes));
		}
		start = end +1 ;
	}
    return result;
}
