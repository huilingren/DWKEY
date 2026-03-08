from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph
from graph cimport Graph
import gc
from libc.stdlib cimport free

cdef class py_Utils:
    cdef shared_ptr[Utils] inner_Utils
    cdef shared_ptr[Graph] inner_Graph
    def __cinit__(self):
        self.inner_Utils = shared_ptr[Utils](new Utils())
    # def __dealloc__(self):
    #     if self.inner_Utils != NULL:
    #         self.inner_Utils.reset()
    #         gc.collect()

    def getRobustness(self,_g,solution):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).in_nodes_weight=_g.in_nodes_weight
        deref(self.inner_Graph).out_nodes_weight=_g.out_nodes_weight
        deref(self.inner_Graph).betweenness=_g.betweenness
        deref(self.inner_Graph).in_total_nodes_weight=_g.in_total_nodes_weight
        deref(self.inner_Graph).out_total_nodes_weight=_g.out_total_nodes_weight
        deref(self.inner_Graph).total_betweenness=_g.total_betweenness
        deref(self.inner_Graph).edge_weights=_g.edge_weights
        deref(self.inner_Graph).edge_weight_sum=_g.edge_weight_sum
        deref(self.inner_Graph).edgetotalweight=_g.edgetotalweight
        deref(self.inner_Graph).in_edge_weight_sum=_g.in_edge_weight_sum
        return deref(self.inner_Utils).getRobustness(self.inner_Graph,solution)


    def reInsert(self,_g,solution,allVex,int decreaseStrategyID,int reinsertEachStep):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).in_nodes_weight=_g.in_nodes_weight
        deref(self.inner_Graph).out_nodes_weight=_g.out_nodes_weight
        deref(self.inner_Graph).betweenness=_g.betweenness
        deref(self.inner_Graph).in_total_nodes_weight=_g.in_total_nodes_weight
        deref(self.inner_Graph).out_total_nodes_weight=_g.out_total_nodes_weight
        deref(self.inner_Graph).total_betweenness=_g.total_betweenness
        deref(self.inner_Graph).edge_weights=_g.edge_weights
        return deref(self.inner_Utils).reInsert(self.inner_Graph,solution,allVex,decreaseStrategyID,reinsertEachStep)

    def getMxWccSz(self, _g):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).in_nodes_weight=_g.in_nodes_weight
        deref(self.inner_Graph).out_nodes_weight=_g.out_nodes_weight
        deref(self.inner_Graph).betweenness=_g.betweenness
        deref(self.inner_Graph).in_total_nodes_weight=_g.in_total_nodes_weight
        deref(self.inner_Graph).out_total_nodes_weight=_g.out_total_nodes_weight
        deref(self.inner_Graph).total_betweenness=_g.total_betweenness
        deref(self.inner_Graph).edge_weights=_g.edge_weights
        return deref(self.inner_Utils).getMxWccSz(self.inner_Graph)

    def Betweenness(self,_g):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        return deref(self.inner_Utils).Betweenness(self.inner_Graph)

    @property
    def MaxWccSzList(self):
        return deref(self.inner_Utils).MaxWccSzList

    @property
    def CostRatios(self):
        return deref(self.inner_Utils).CostRatios

    @property
    def CostRatios1(self):
        return deref(self.inner_Utils).CostRatios1

    @property
    def covered_set(self):
        return deref(self.inner_Utils).covered_set
