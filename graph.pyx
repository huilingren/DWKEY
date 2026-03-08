'''
#file:graph.pyx类graph的实现文件
#可以自动导入相同路径下相同名称的.pxd的文件
#可以省略cimport graph命令
#需要重新设计python调用的接口，此文件
'''
from cython.operator cimport dereference as deref
cimport cpython.ref as cpy_ref
from libcpp.memory cimport shared_ptr
from libc.stdlib cimport malloc
from libc.stdlib cimport free
import numpy as np

cdef class py_Graph:
    cdef shared_ptr[Graph] inner_graph#使用unique_ptr优于shared_ptr
    #__cinit__会在__init__之前被调用
    def __cinit__(self,*arg):
        '''doing something before python calls the __init__.
        cdef 的C/C++对象必须在__cinit__里面完成初始化，否则没有为之分配内存
        可以接收参数，使用python的变参数模型实现类似函数重载的功能。'''
        #print("doing something before python calls the __init__")
        # if len(arg)==0:
        #     print("num of parameter is 0")
        self.inner_graph = shared_ptr[Graph](new Graph())
        cdef int _num_nodes
        cdef int _num_edges
        cdef int[:] edges_from
        cdef int[:] edges_to
        cdef double[:] in_nodes_weight
        cdef double[:] out_nodes_weight
        cdef double[:] betweenness
        cdef double[:] edge_weights
        #print(f"Received arguments: {arg}")
        #print("arg[6] content:", arg[6])
        #print("[CYTHON] 收到参数数量:", len(arg))  # 检查是否收到8个参数
        if len(arg)==0:
            #这两行代码为了防止内存没有初始化，没有实际意义
            deref(self.inner_graph).num_edges=0
            deref(self.inner_graph).num_nodes=0
        elif len(arg)==8:
            _num_nodes=arg[0]
            _num_edges=arg[1]
            #print("arg[6] = ", arg[6])

            edges_from = np.array([int(x) for x in arg[2]], dtype=np.int32)
            edges_to = np.array([int(x) for x in arg[3]], dtype=np.int32)
            in_nodes_weight = np.array([x for x in arg[4]], dtype=np.double)
            out_nodes_weight = np.array([x for x in arg[5]], dtype = np.double)
            betweenness = np.array([x for x in arg[6]], dtype=np.double)
            edge_weights = np.array([x for x in arg[7]], dtype=np.double)
            self.reshape_Graph(_num_nodes, _num_edges, edges_from, edges_to, in_nodes_weight, out_nodes_weight, betweenness, edge_weights)
        else:
            print(f"Error：py_Graph类未被成功初始化，因为提供参数数目不匹配,参数个数为{len(arg)}，参数个数为0或8。")


    @property
    def num_nodes(self):
        return deref(self.inner_graph).num_nodes

    # @num_nodes.setter
    # def num_nodes(self):
    #     def __set__(self,num_nodes):
    #         self.setadj(adj_list)

    @property
    def num_edges(self):
        return deref(self.inner_graph).num_edges

    @property
    def in_total_nodes_weight(self):
        return deref(self.inner_graph).in_total_nodes_weight

    @property
    def out_total_nodes_weight(self):
        return deref(self.inner_graph).out_total_nodes_weight

    @property
    def total_betweenness(self):
        return deref(self.inner_graph).total_betweenness

    @property
    def in_nodes_weight(self):
        return deref(self.inner_graph).in_nodes_weight

    @property
    def out_nodes_weight(self):
        return deref(self.inner_graph).out_nodes_weight

    @property
    def betweenness(self):
        return deref(self.inner_graph).betweenness

    @property
    def adj_list(self):
        return deref(self.inner_graph).adj_list

    @property
    def edge_list(self):
        return deref(self.inner_graph).edge_list

    @property
    def edge_weights(self):
        return deref(self.inner_graph).edge_weights

    @property
    def edgetotalweight(self):
        return deref(self.inner_graph).edgetotalweight
    
    @property
    def edge_weight_sum(self):
        return deref(self.inner_graph).edge_weight_sum
    
    @property
    def in_edge_weight_sum(self):
        return deref(self.inner_graph).in_edge_weight_sum

    cdef reshape_Graph(self, int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to,double[:] in_nodes_weight,double[:] out_nodes_weight,double[:] betweenness, double[:] edge_weights):
        cdef int *cint_edges_from = <int*>malloc(_num_edges*sizeof(int))
        cdef int *cint_edges_to = <int*>malloc(_num_edges*sizeof(int))
        cdef double *cdoube_in_nodes_weight = <double*>malloc(_num_nodes*sizeof(double))
        cdef double *cdoube_out_nodes_weight = <double*>malloc(_num_nodes*sizeof(double))
        cdef double *cdoube_betweenness = <double*>malloc(_num_nodes*sizeof(double))
        cdef double *cdoube_edge_weights = <double*>malloc(_num_edges*sizeof(double))
        cdef int i
        for i in range(_num_nodes):
            cdoube_in_nodes_weight[i] = in_nodes_weight[i]
        for i in range(_num_nodes):
            cdoube_out_nodes_weight[i] = out_nodes_weight[i]
        for i in range(_num_nodes):
            cdoube_betweenness[i] = betweenness[i]
        for i in range(_num_edges):
            cint_edges_from[i] = edges_from[i]
        for i in range(_num_edges):
            cint_edges_to[i] = edges_to[i]
        for i in range(_num_edges):
            cdoube_edge_weights[i] = edge_weights[i]

        self.inner_graph = shared_ptr[Graph](new Graph(_num_nodes,_num_edges,&cint_edges_from[0],&cint_edges_to[0],&cdoube_in_nodes_weight[0],&cdoube_out_nodes_weight[0],&cdoube_betweenness[0],&cdoube_edge_weights[0]))
        free(cint_edges_from)
        free(cint_edges_to)

    def reshape(self,int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to,double[:] in_nodes_weight,double[:] out_nodes_weight,double[:] betweenness,double[:] edge_weights):
        self.reshape_Graph(_num_nodes, _num_edges, edges_from, edges_to,in_nodes_weight,out_nodes_weight,betweenness,edge_weights)


cdef class py_GSet:
    cdef shared_ptr[GSet] inner_gset
    def __cinit__(self):
        self.inner_gset = shared_ptr[GSet](new GSet())
    def InsertGraph(self,int gid,py_Graph graph):
        deref(self.inner_gset).InsertGraph(gid,graph.inner_graph)
        #self.InsertGraph(gid,graph.inner_graph)

        # deref(self.inner_gset).InsertGraph(gid,graph.inner_graph)
         #self.Inner_InsertGraph(gid,graph.inner_graph)

    def Sample(self):
        temp_innerGraph=deref(deref(self.inner_gset).Sample())   #得到了Graph 对象
        return self.G2P(temp_innerGraph)

    def Get(self,int gid):
        temp_innerGraph=deref(deref(self.inner_gset).Get(gid))   #得到了Graph 对象
        return self.G2P(temp_innerGraph)

    def Clear(self):
        deref(self.inner_gset).Clear()

    cdef G2P(self,Graph graph):
        num_nodes = graph.num_nodes     #得到Graph对象的节点个数
        num_edges = graph.num_edges    #得到Graph对象的连边个数
        edge_list = graph.edge_list
        in_nodes_weight = graph.in_nodes_weight
        out_nodes_weight = graph.out_nodes_weight
        betweenness = graph.betweenness
        edge_weights = graph.edge_weights

        cint_edges_from = np.zeros([num_edges],dtype=int)
        cint_edges_to = np.zeros([num_edges],dtype=int)
        cdouble_in_nodes_weight = np.zeros([num_nodes],dtype=np.double)
        cdouble_out_nodes_weight = np.zeros([num_nodes],dtype=np.double)
        cdouble_betweenness = np.zeros([num_nodes],dtype=np.double)
        cdouble_edge_weights = np.zeros([num_edges],dtype=np.double)

        cdef int i
        for i in range(num_nodes):
            cdouble_in_nodes_weight[i] = in_nodes_weight[i]
            cdouble_out_nodes_weight[i] = out_nodes_weight[i]
            cdouble_betweenness[i] = betweenness[i]
        for i in range(num_edges):
            cint_edges_from[i]=edge_list[i].first
            cint_edges_to[i] =edge_list[i].second
            cdouble_edge_weights[i] =edge_weights[i]
        return py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to,cdouble_in_nodes_weight,cdouble_out_nodes_weight,cdouble_betweenness,cdouble_edge_weights)


