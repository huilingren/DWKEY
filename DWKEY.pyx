
from __future__ import print_function, division
import tensorflow.compat.v1 as tf  
tf.disable_v2_behavior()          

import numpy as np
import networkx as nx
import random
import time
import pickle as cp
import sys
from tqdm import tqdm
import PrepareBatchGraph
import graph
import nstep_replay_mem
import nstep_replay_mem_prioritized
import mvc_env
import utils
import heapq
import scipy.linalg as linalg
import os
import pandas as pd

# Hyper Parameters:
cdef double GAMMA = 0.99  # decay rate of past observations 1
cdef int UPDATE_TIME = 1000
cdef int EMBEDDING_SIZE = 128  # 64
cdef int MAX_ITERATION = 1000000
cdef double LEARNING_RATE = 0.0001   #学习率 0.0001 
cdef int MEMORY_SIZE = 5000000
cdef double Alpha = 0.001 ## weight of reconstruction loss  0.001
########################### hyperparameters for priority(start)#########################################
cdef double epsilon = 0.0000001  # small amount to avoid zero priority
cdef double alpha = 0.6  # [0~1] convert the importance of TD error to priority
cdef double beta = 0.4  # importance-sampling, from initial value increasing to 1
cdef double beta_increment_per_sampling = 0.001
cdef double TD_err_upper = 1.  # clipped abs error
########################## hyperparameters for priority(end)#########################################
cdef int N_STEP = 5
cdef int NUM_MIN = 30 #30
cdef int NUM_MAX = 50 #50
cdef int REG_HIDDEN = 32 # 32
cdef int M = 4  # how many edges selected each time for BA model
cdef int BATCH_SIZE = 128  #64
cdef double initialization_stddev = 0.01  # 权重初始化的方差 0.01
cdef int n_valid = 200
cdef int aux_dim = 5  # 全局特征
cdef int num_env = 1
cdef double inf = 2147483647/2
#########################  embedding method ##########################################################
cdef int max_bp_iter = 2
cdef int aggregatorID = 0 #0:sum; 1:mean; 2:GCN
cdef int embeddingMethod = 1   #0:structure2vec; 1:graphsage; 2:attention


class DWKEY:

    def __init__(self):
        # init some parameters
        self.embedding_size = EMBEDDING_SIZE
        self.learning_rate = LEARNING_RATE
        self.g_type = 'barabasi_albert'#erdos_renyi, powerlaw, small-world,barabasi_albert,non_normal_network,CM
        self.training_type = 'random'   #'random'
        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.inputs = dict()
        self.reg_hidden = REG_HIDDEN
        self.utils = utils.py_Utils()
        
        ############----------------------------- variants of DQN(start) ------------------- ###################################
        self.IsHuberloss = False
        self.IsDoubleDQN = False
        
        self.IsPrioritizedSampling = False
        self.IsDuelingDQN = False
        self.IsMultiStepDQN = True     ##(if IsNStepDQN=False, N_STEP==1)
        self.IsDistributionalDQN = False
        self.IsNoisyNetDQN = False
        self.Rainbow = False
        ############----------------------------- variants of DQN(end) ------------------- ###################################

        #Simulator
        self.ngraph_train = 0
        self.ngraph_test = 0
        self.env_list=[]
        self.g_list=[]
        # self.covered=[]
        self.pred=[]
        if self.IsPrioritizedSampling:
            self.nStepReplayMem = nstep_replay_mem_prioritized.py_Memory(epsilon,alpha,beta,beta_increment_per_sampling,TD_err_upper,MEMORY_SIZE)
        else:
            self.nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(MEMORY_SIZE)

        for i in range(num_env):
            #print(f"Processing graph {i + 1}")
            self.env_list.append(mvc_env.py_MvcEnv(NUM_MAX))
            self.g_list.append(graph.py_Graph())

        self.test_env = mvc_env.py_MvcEnv(NUM_MAX)

        # [batch_size, node_cnt]
        self.action_select = tf.sparse_placeholder(tf.float32, name="action_select")
        # [node_cnt, batch_size]
        self.rep_global = tf.sparse_placeholder(tf.float32, name="rep_global")
        # [node_cnt, node_cnt]
        self.n2nsum_param = tf.sparse_placeholder(tf.float32, name="n2nsum_param")
        # [node_cnt, node_cnt]
        self.laplacian_param = tf.sparse_placeholder(tf.float32, name="laplacian_param")
        # [batch_size, node_cnt]
        self.subgsum_param = tf.sparse_placeholder(tf.float32, name="subgsum_param")
        # [batch_size,1]
        self.target = tf.placeholder(tf.float32, [BATCH_SIZE,1], name="target")
        # [batch_size, aux_dim]
        self.aux_input = tf.placeholder(tf.float32, name="aux_input")
        # [node_cnt, 2]
        self.node_input = tf.placeholder(tf.float32, [None, 2], name="node_input")
        #self.node_input = tf.placeholder(tf.float32, name="node_input")  原来的


        #[batch_size, 1]
        if self.IsPrioritizedSampling:
            self.ISWeights = tf.placeholder(tf.float32, [BATCH_SIZE, 1], name='IS_weights')


        # init Q network
        self.loss,self.trainStep,self.loss_recons,self.q_pred, self.q_on_all,self.Q_param_list = self.Encode() 
        #init Target Q Network
        self.lossT,self.trainStepT,self.loss_recons,self.q_predT, self.q_on_allT,self.Q_param_listT = self.Encode()
        #takesnapsnot
        self.copyTargetQNetworkOperation = [a.assign(b) for a,b in zip(self.Q_param_listT,self.Q_param_list)]


        self.UpdateTargetQNetwork = tf.group(*self.copyTargetQNetworkOperation)
        # saving and loading networks
        self.saver = tf.train.Saver(max_to_keep=None)
        #self.session = tf.InteractiveSession()
        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=100,
                                intra_op_parallelism_threads=100,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config = config)

        # self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
        self.session.run(tf.global_variables_initializer())


########################################## ENCODE FOR DIRECTED #####################################################

    def Encode(self):  
        # 初始化权重矩阵
        w_n2l = tf.Variable(tf.truncated_normal([2, self.embedding_size], stddev=initialization_stddev), tf.float32)
        # [embed_dim, embed_dim] 将当前层嵌入向量映射到下一层的权重矩阵
        p_node_conv_r = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
        p_node_conv_s = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
        p_node_conv_v = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
        if embeddingMethod == 1:    #'graphsage' 增加额外的权重矩阵，进行更复杂的消息传递和特征更新
            # [embed_dim, embed_dim]
            p_node_conv2 = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
            p_node_conv2_r = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
            p_node_conv2_s = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
            # [2*embed_dim, embed_dim]
            p_node_conv3_r = tf.Variable(tf.truncated_normal([2*self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
            p_node_conv3_s = tf.Variable(tf.truncated_normal([2*self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)
            p_node_conv4 = tf.Variable(tf.truncated_normal([2*self.embedding_size, self.embedding_size], stddev=initialization_stddev), tf.float32)

        #定义用于回归层的权重：
        #h1_weight 用于将嵌入映射到隐藏层（如果有）。
        #h2_weight 用于将隐藏层（加上辅助维度）映射到最终的输出
        #[reg_hidden+aux_dim, 1]
        # reg_hidden = 30
        if self.reg_hidden > 0:
            h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden + aux_dim, 1], stddev=initialization_stddev), tf.float32)
            last_w = h2_weight
        else:
            h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            last_w = h1_weight

        ## [embed_dim, 1] cross_product 是一个额外的权重，用于计算节点嵌入的交叉特征
        cross_product = tf.Variable(tf.truncated_normal([self.embedding_size, 1], stddev=initialization_stddev), tf.float32)

        y_nodes_size = tf.shape(self.subgsum_param)[0]  # 子图中节点的数量
        # [batch_size, 2]
        y_node_input = tf.ones((y_nodes_size,2))  # 全为 1 的占位符张量，每个节点的初始特征
 
        #[node_cnt, 2] * [2, embed_dim] = [node_cnt, embed_dim]
        # 生成节点消息：通过矩阵乘法将节点的初始特征 (node_input) 与权重矩阵 w_n2l 相乘，得到每个节点的消息表示 input_message
        # 接着通过 ReLU 激活函数得到节点的潜在表示 input_potential_layer，表示节点的潜在嵌入。
        # no sparse
        input_message = tf.matmul(tf.cast(self.node_input,tf.float32), w_n2l)  # 将节点初始特征通过矩阵乘法映射到嵌入空间
        #[node_cnt, embed_dim]  # no sparse
        input_potential_layer = tf.nn.relu(input_message) # 引入非线性

        # # no sparse
        # [batch_size, embed_dim]
        y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
        #[batch_size, embed_dim]  # no sparse
        y_input_potential_layer = tf.nn.relu(y_input_message)

        input_potential_layer = input_message
        cdef int lv = 0
        #[node_cnt, embed_dim], no sparse  节点消息归一化，将生成的节点消息（input_potential_layer）按行进行 L2 归一化，确保每个节点的消息在嵌入空间中的尺度一致。
        cur_message_layer = input_potential_layer
        cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        # [batch_size, embed_dim]
        y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
       
        # 消息传递的过程通过一个迭代循环（while lv < max_bp_iter）进行。在每一次迭代中，节点从其邻居节点接收消息并进行聚合。这种迭代的过程使得每个节点能够逐渐获得来自更远距离节点的信息。
        # max_bp_iter = 2 节点传播两阶
        while lv < max_bp_iter:
            lv = lv + 1
            # self.n2nsum_param: [num_nodes, num_nodes] (稀疏)
            n2npool_r = tf.sparse_tensor_dense_matmul(tf.cast(self.n2nsum_param,tf.float32), cur_message_layer) # [num_nodes, E]
            n2nsum_param_t= tf.sparse.transpose(self.n2nsum_param)
            n2npool_s = tf.sparse_tensor_dense_matmul(tf.cast(n2nsum_param_t,tf.float32), cur_message_layer) # [num_nodes, E]

            node_linear_r = tf.matmul(n2npool_r, p_node_conv_r) # [num_nodes, E]
            node_linear_s = tf.matmul(n2npool_s, p_node_conv_s) # [num_nodes, E]

            # cur_message_layer: [num_nodes, embed_dim]
            subgraph_features_pooled = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer) # [num_subgraph, E]

            # self.subgsum_param 的转置 (subgsum_param_t) 形状是 [num_nodes, num_subgraph]
            subgsum_param_t = tf.sparse.transpose(self.subgsum_param) # [num_nodes, num_subgraph]

            # 结果: [num_nodes, embed_dim]
            nodes_features_from_subgraphs = tf.sparse_tensor_dense_matmul(tf.cast(subgsum_param_t,tf.float32), y_cur_message_layer) # [num_nodes, E]


            if embeddingMethod == 1: # 'graphsage'
                cur_message_layer_linear_r = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2_r) # [num_nodes, E]
                cur_message_layer_linear_s = tf.matmul(tf.cast(cur_message_layer, tf.float32), p_node_conv2_s) # [num_nodes, E]

                merged_linear_r = tf.concat([node_linear_r, cur_message_layer_linear_r], 1) # [num_nodes, 2*E]
                merged_linear_s = tf.concat([node_linear_s, nodes_features_from_subgraphs], 1) # [num_nodes, 2*E]

                cur_message_layer_r = tf.nn.relu(tf.matmul(merged_linear_r, p_node_conv3_r)) # [num_nodes, 2*E] * [2*E, E] = [num_nodes, E]
                cur_message_layer_s = tf.nn.relu(tf.matmul(merged_linear_s, p_node_conv3_s)) # [num_nodes, 2*E] * [2*E, E] = [num_nodes, E]

                y_cur_message_layer_linear_r = tf.matmul(tf.cast(subgraph_features_pooled, tf.float32), p_node_conv2_r) # [num_subgraph, E] * [E, E] = [num_subgraph, E]
                y_cur_message_layer_linear_s = tf.matmul(tf.cast(subgraph_features_pooled, tf.float32), p_node_conv2_s) # [num_subgraph, E] * [E, E] = [num_subgraph, E]

                y_merged_linear_r = tf.concat([subgraph_features_pooled, y_cur_message_layer_linear_r], 1) # [num_subgraph, 2*E]
                y_merged_linear_s = tf.concat([subgraph_features_pooled, y_cur_message_layer_linear_s], 1) # [num_subgraph, 2*E]

                y_cur_message_layer_r = tf.nn.relu(tf.matmul(y_merged_linear_r, p_node_conv3_r)) # [num_subgraph, 2*E] * [2*E, E] = [num_subgraph, E]
                y_cur_message_layer_s = tf.nn.relu(tf.matmul(y_merged_linear_s, p_node_conv3_s)) # [num_subgraph, 2*E] * [2*E, E] = [num_subgraph, E]

            y_cur_message_layer_all = tf.concat([y_cur_message_layer_r, y_cur_message_layer_s], 1) # [num_subgraph, 2*E]
            y_cur_message_layer = tf.nn.relu(tf.matmul(y_cur_message_layer_all, p_node_conv4)) # [num_subgraph, 2*E] * [2*E, E] = [num_subgraph, E]
            y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1) # [num_subgraph, E]

            # 更新 cur_message_layer 用于下一轮循环
            cur_message_layer_all = tf.concat([cur_message_layer_r,cur_message_layer_s], 1) # [num_nodes, 2*E]
            cur_message_layer = tf.nn.relu(tf.matmul(cur_message_layer_all,p_node_conv4)) # [num_nodes, 2*E] * [2*E, E] = [num_nodes, E]
            cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1) # [num_nodes, E]
        # 得到最终节点嵌入cur_message_layer，融入了邻居信息的节点表示
        self.node_embedding = cur_message_layer  # 每个节点的表示会通过该消息层来初始化
        y_potential = y_cur_message_layer  # 代表全局状态
        action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_message_layer) # 表示 batch 中每个样本经过动作选择后的嵌入
        temp = tf.matmul(tf.expand_dims(action_embed, axis=2),tf.expand_dims(y_potential, axis=1)) # 将状态和动作嵌入的信息以矩阵的形式进行交互建模
        # [batch_size, embed_dim]
        Shape = tf.shape(action_embed)
        # [batch_size, embed_dim], first transform
        embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product,[Shape[0],1]),[Shape[0],Shape[1],1])),Shape) # 生成表示状态-动作交互的向量 embed_s_a， 将状态信息（y_potential）和动作信息（action_embed）通过 temp 中的交互矩阵与 cross_product 参数进行融合
        
        #[batch_size, 2 * embed_dim]
        last_output = embed_s_a

        # reg_hidden = 30
        if self.reg_hidden > 0:
            #[batch_size, 2*embed_dim] * [2*embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            hidden = tf.matmul(embed_s_a, h1_weight)
            #[batch_size, reg_hidden]
            last_output = tf.nn.relu(hidden)
        last_output = tf.concat([last_output, self.aux_input], 1)
        q_pred = tf.matmul(last_output, last_w) # 预测Q值
        
        loss_recons = 2 * tf.trace(tf.matmul(tf.transpose(cur_message_layer), tf.sparse_tensor_dense_matmul(tf.cast(self.laplacian_param,tf.float32), cur_message_layer)))
        
        edge_num = tf.sparse_reduce_sum(self.n2nsum_param)   # 入边出边相等，只统计一次
        loss_recons = tf.divide(loss_recons, edge_num)

        if self.IsPrioritizedSampling:
            self.TD_errors = tf.reduce_sum(tf.abs(self.target - q_pred), axis=1)    # for updating Sumtree
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.ISWeights * self.target, self.ISWeights * q_pred)
            else:
                loss_rl = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target, q_pred))
        else:
            if self.IsHuberloss:
                loss_rl = tf.losses.huber_loss(self.target, q_pred)
            else:
                loss_rl = tf.losses.mean_squared_error(self.target, q_pred) # pick this
        loss = loss_rl + Alpha * loss_recons

        trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        #[node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
        rep_y = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), y_potential)
        temp1 = tf.matmul(tf.expand_dims(cur_message_layer, axis=2),tf.expand_dims(rep_y, axis=1))
        # [node_cnt embed_dim]
        Shape1 = tf.shape(cur_message_layer)
        embed_s_a_all = tf.reshape(tf.matmul(temp1, tf.reshape(tf.tile(cross_product,[Shape1[0],1]),[Shape1[0],Shape1[1],1])),Shape1)

        #[node_cnt, 2 * embed_dim]
        last_output = embed_s_a_all
        if self.reg_hidden > 0:
            #[node_cnt, 2 * embed_dim] * [2 * embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
            hidden = tf.matmul(embed_s_a_all, h1_weight)
            #Relu, [node_cnt, reg_hidden1]
            last_output = tf.nn.relu(hidden)

        #[node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
        rep_aux = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), self.aux_input)

        last_output = tf.concat([last_output,rep_aux],1)

        q_on_all = tf.matmul(last_output, last_w)
        return loss,trainStep,loss_recons,q_pred,q_on_all,tf.trainable_variables()
   
   

    # 有向图生成模型
    def non_normal_network(self, N, N0, m, p):
        """
        生成一个有向非正态网络（Non-Normal Network）

        参数：
        N  - 总节点数
        N0 - 初始完全连接子图中的节点数
        m  - 每个新加入节点的连接数
        p  - 反转边的概率

        返回：
        G  - 生成的有向图（DiGraph）
        """
        G = nx.DiGraph()  # 创建一个有向图
        G.add_nodes_from(range(0, N))  # 添加N个节点

        edges = []  # 存储边信息
        k = 0  # 当前边数索引

        # 构造初始完全连接子图
        if N0 < m + 1:
            for i in range(N0 + 1, m + 2):
                for j in range(1, i):
                    edges.append((j, i))
                    k += 1

        # 逐步增加节点并进行连接
        for i in range(m + 2, N + 1):
            degrees = np.array([sum(1 for edge in edges if edge[0] == j) for j in range(1, i)])
            degrees += 1  # 防止零概率情况

            # 按度数分布加权随机选择 m 个不同的节点
            probs = degrees / degrees.sum()
            neighbors = np.random.choice(range(1, i), size=m, replace=False, p=probs)

            for neighbor in neighbors:
                edges.append((neighbor, i))
                k += 1

        # 反转部分边
        if p > 0:
            edge_count = len(edges)
            flip_indices = np.where(np.random.rand(edge_count) < p)[0]  # 选择需要翻转的边索引
            for idx in flip_indices:
                u, v = edges[idx]
                if (v, u) not in edges and u != v:  # 避免自环和多重边
                    edges[idx] = (v, u)  # 直接反转原边
        # 添加边到图
        G.add_edges_from(edges)

        return G  # 返回构造的有向图

    def generate_directed_configuration_model(self, N, m, lambda_param, q):
        # 生成符合幂律分布的度序列
        alpha = lambda_param - 1  # Paret分布的形状参数
        samples = (np.random.pareto(alpha, N) + 1) * m  # 生成样本并调整最小值为m
        degrees = np.floor(samples).astype(int)  # 转换为整数度数
        degrees = np.clip(degrees, m, N-1)  # 限制度数范围
        
         # 调整总度数为偶数（避免度数越界）
        total_degree = np.sum(degrees)
        if total_degree % 2 != 0:
            adjusted = False
            # 寻找第一个可安全调整的节点（度数 < N-1）
            for i in range(len(degrees)):
                if degrees[i] < N-1:
                    degrees[i] += 1
                    adjusted = True
                    break
            if not adjusted:
                raise ValueError("无法调整总度数为偶数：所有节点的度数已达上限(N-1)")
        
        # 生成配置模型的无向图
        G = nx.configuration_model(degrees)
        G = nx.Graph(G)  # 转换为简单图（去除多边）
        G.remove_edges_from(nx.selfloop_edges(G))  # 移除自环边
        

        # 转换为有向图
        D = nx.DiGraph()
        D.add_nodes_from(range(N))  # 强制确保节点数为N
        for u, v in G.edges():
            if np.random.rand() < q:
                # 随机选择单向边方向
                if np.random.rand() < 0.5:
                    D.add_edge(u, v)
                else:
                    D.add_edge(v, u)
            else:
                # 添加双向边
                D.add_edge(u, v)
                D.add_edge(v, u)
        
        return D

    def gen_graph(self, num_min, num_max):
        cdef int max_n = num_max
        cdef int min_n = num_min
        cdef int cur_n = np.random.randint(max_n - min_n + 1) + min_n
        
        if self.g_type == 'non_normal_network':
            g = self.non_normal_network(N=cur_n, N0=5, m=4, p=0.1)
        elif self.g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
        elif self.g_type == 'powerlaw':
            g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
        elif self.g_type == 'small-world':
            g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
        elif self.g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=cur_n, m=4)
        elif self.g_type == 'directed':
            g = nx.scale_free_graph(n=cur_n, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0)
        elif self.g_type == 'CM':
            g = self.generate_directed_configuration_model(N = cur_n, m = 2, lambda_param = 3, q = 0.3)
        g = nx.DiGraph(g)

        
        # 添加边权重
        if self.training_type == 'random':
            for u, v in g.edges():
                g[u][v]['weight'] = random.uniform(0, 1)
        else:
            # 幂律分布的权重，范围（1，无穷）
            alpha = 2.5  # 形状参数，通常在 (2, 3) 之间
            for u, v in g.edges():
                g.edges[u][v]['weight'] = np.random.pareto(alpha) + 1  # 使最小权重不为 0
        
        # 基于边权计算节点权重
        btw = {}
        for node in g.nodes():  # This is the crucial fix - iterate through nodes first
            in_weight_sum = sum(
                float(g[u][node]['weight'])  # 显式转换为float
                for u in g.predecessors(node)
            )
            # 出边权重和
            out_weight_sum = sum(
                float(g[node][v]['weight']) 
                for v in g.successors(node)
            )
            btw[node] = in_weight_sum + out_weight_sum
    
        # 归一化
        max_btw = max(btw.values()) if btw.values() else 1
        for node in btw:
            btw[node] /= max_btw
        nx.set_node_attributes(g, btw, 'btw')
        in_degree = dict(g.in_degree())
        out_degree = dict(g.out_degree())
        nx.set_node_attributes(g, in_degree, 'in_degree')
        nx.set_node_attributes(g, out_degree, 'out_degree')
        return g 

    def gen_new_graphs(self, num_min, num_max):
        print('\ngenerating new training graphs...')
        sys.stdout.flush()
        self.ClearTrainGraphs()
        cdef int i
        for i in tqdm(range(1000)):
            g = self.gen_graph(num_min, num_max)
            self.InsertGraph(g, is_test=False)

    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()

    def InsertGraph(self,g,is_test):
        cdef int t
        #print(f"Graph before insertion: {g}")  
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, self.GenNetwork(g))
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, self.GenNetwork(g))

    #pass
    def PrepareValidData(self):
        print('\ngenerating validation graphs...')
        sys.stdout.flush()
        cdef double result_degree = 0.0
        cdef double result_betweeness = 0.0
        # n_valid = 200 这里计算鲁棒性是为了和之后进行对比，这里的鲁棒性是根据相应的方法得到的，我们是用深度强化学习得到的
        for i in tqdm(range(n_valid)):
            g = self.gen_graph(NUM_MIN, NUM_MAX)
            g_degree = g.copy()
            g_betweenness = g.copy()
            result_degree += self.HXA(g_degree,'HDA')
            result_betweeness += self.HXA(g_betweenness, 'HBA')
            self.InsertGraph(g, is_test=True)
        print ('Validation of HDA: %.16f'%(result_degree / n_valid))
        print ('Validation of HBA: %.16f'%(result_betweeness / n_valid))

    def Run_simulator(self, int num_seq, double eps, TrainSet, int n_step):
        # num_seq = 100 ; eps = 1  模拟智能体与环境交互逻辑
        cdef int num_env = len(self.env_list)  # 计算环境数量，包含多个环境实例
        # 初始化序列计数器
        cdef int n = 0  # 初始化计数器，表示当前已经生成的序列数量
        cdef int i 
        #print ("Run_simulator is working")
        # 当序列计数器小于所需序列数量10时循环
        while n < num_seq:
            for i in range(num_env): # 遍历所有环境，逐一检查并与之交互
                if self.env_list[i].graph.num_nodes == 0 or self.env_list[i].isTerminal():
                    if self.env_list[i].graph.num_nodes > 0 and self.env_list[i].isTerminal():
                        n = n + 1
                        self.nStepReplayMem.Add(self.env_list[i], n_step) # 将当前的环境状态添加到里面，用于后续训练
                    g_sample= TrainSet.Sample() # 随机采样一个图数据作为环境初始状态
                    self.env_list[i].s0(g_sample)
                    self.g_list[i] = self.env_list[i].graph 
            if n >= num_seq:
                break # 退出循环
            Random = False
            if random.uniform(0,1) >= eps:
                pred = self.PredictWithCurrentQNet(self.g_list, [env.action_list for env in self.env_list])  # 利用当前策略采取动作
            else:
                Random = True
            for i in range(num_env):
                if (Random):
                    a_t = self.env_list[i].randomAction()  # 随机探索动作
                else:
                    a_t = self.argMax(pred[i]) # 基于策略选取Q值最大的策略
                self.env_list[i].step(a_t) # 在环境中选定动作，更新环境状态
    #pass
    def PlayGame(self,int n_traj, double eps):
        self.Run_simulator(n_traj, eps, self.TrainSet, N_STEP)  # 主要是通过这个玩游戏产生四元组，放到经验回放池中，供后面模型训练


    def SetupTrain(self, idxes, g_list, covered, actions, target):
        self.m_y = target
        self.inputs['target'] = self.m_y
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)
        self.inputs['action_select'] = prepareBatchGraph.act_select
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['node_input'] = prepareBatchGraph.node_feat
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat


    def SetupPredAll(self, idxes, g_list, covered):
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupPredAll(idxes, g_list, covered)
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        # self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['node_input'] = prepareBatchGraph.node_feat
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat
        return prepareBatchGraph.idx_map_list

    def Predict(self,g_list,covered,isSnapSnot):
        # 计算每个节点的预测值
        cdef int n_graphs = len(g_list)  # 图的数量 
        cdef int i, j, k, bsize # i 循环迭代、j节点处理、k和bsize批次大小
        
        #pred = []
        for i in range(0, n_graphs, BATCH_SIZE): # BATCH_SIZE对图列表g_list进行分批处理。i是当前图的起始图索引
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs: # 如果剩余的图数量小于BATCH_SIZE
                bsize = n_graphs - i  # 将当前批次大小设置为剩余图的数量
            batch_idxes = np.zeros(bsize)  # 存储当前批次中所有图的索引
            for j in range(i, i + bsize):  
                batch_idxes[j - i] = j  # 遍历每个图的索引并将其存储在batch_idxes中
            batch_idxes = np.int32(batch_idxes)  
            #print(f"Processing batch_idxes: {batch_idxes}")  # 检查当前批次的图索引
            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)  # SetupPredAll 会根据图的索引和已覆盖的节点，返回一个 idx_map_list，每个元素是一个图的节点映射
            # 将列表转换为 NumPy 数组或 TensorFlow 张量后再打印形状
            node_input_array = np.array(self.inputs['node_input'])
           # print(f"Shape of node_input: {node_input_array.shape}")
            #print(f"Shape of node_input: {node_input_array.shape}")
            if node_input_array.shape[0] == 0:
                print(f"Skipping graph {i} as node_input is empty.")
              #  continue  # 跳过这个图
            #node_input = np.array(self.inputs['node_input'])
            #if node_input.size == 0:
            #    continue

            if isSnapSnot:
                result = self.session.run([self.q_on_allT], feed_dict={
                    self.rep_global: self.inputs['rep_global'],
                    self.n2nsum_param: self.inputs['n2nsum_param'],
                    self.subgsum_param: self.inputs['subgsum_param'],
                    #self.node_input: self.inputs['node_input'],
                    self.node_input: np.array(self.inputs['node_input']).reshape(-1, 2),
                    self.aux_input: np.array(self.inputs['aux_input']),
                })
            else:
                result = self.session.run([self.q_on_all], feed_dict={
                    self.rep_global: self.inputs['rep_global'],
                    self.n2nsum_param: self.inputs['n2nsum_param'],
                    self.subgsum_param: self.inputs['subgsum_param'],
                    self.node_input: np.array(self.inputs['node_input']).reshape(-1, 2),
                    #self.node_input: self.inputs['node_input'],
                    self.aux_input: np.array(self.inputs['aux_input']),
                })
            raw_output = result[0]  # 计算得到的输出，（节点数，1）
            #print("Shape of raw_output:", raw_output.shape)
            pos = 0  # 跟踪raw_output中的当前位置
            pred = []
            for j in range(i, i + bsize): 
                idx_map = idx_map_list[j-i] # 存储当前图节点的映射
                cur_pred = np.zeros(len(idx_map)) # 大小与当前图的节点数相同
                for k in range(len(idx_map)): 
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf # 节点无效，预测值为-inf
                    else:
                        cur_pred[k] = raw_output[pos]  #对有效的节点，赋值给cur_pred
                        pos += 1  # 增加pos
                for k in covered[j]:
                    cur_pred[k] = -inf  # 对已经覆盖的节点设为无效，表示这些节点的预测结果应该被忽略（通常在强化学习中使用，避免重新计算已知的值）
                pred.append(cur_pred)
            assert (pos == len(raw_output))
        # 打印整个预测列表pred
        #print(f"Final pred list: {pred}")
        return pred

    def PredictWithCurrentQNet(self,g_list,covered):
        result = self.Predict(g_list,covered,False)
        return result

    def PredictWithSnapshot(self,g_list,covered):
        result = self.Predict(g_list,covered,True)
        return result
    #pass
    def TakeSnapShot(self):
       self.session.run(self.UpdateTargetQNetwork)

    def Fit(self):
        sample = self.nStepReplayMem.Sampling(BATCH_SIZE)
        ness = False   # 用于标记是否至少存在一个非终止状态，判断是否需要考虑下一状态，如果终止状态则不需要
        cdef int i
        for i in range(BATCH_SIZE):
            if (not sample.list_term[i]):
                ness = True
                break
        if ness:  # 如果存在非终止状态，预测目标Q值 ness =False
            if self.IsDoubleDQN:
                double_list_pred = self.PredictWithCurrentQNet(sample.g_list, sample.list_s_primes)  # 用当前Q网络预测下一状态所有动作的Q值
                double_list_predT = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)  # 用快照（目标Q网络）预测下一状态所有动作的Q值,目标Q网络防止当前Q网络收到当前参数的影响，过度估计使得训练不稳定
                list_pred = [a[self.argMax(b)] for a, b in zip(double_list_predT, double_list_pred)] # 用 double_list_pred 选择最大 Q 值的动作索引，再从 double_list_predT 中获取该动作对应的 Q 值
            else:
                list_pred = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)

        list_target = np.zeros([BATCH_SIZE, 1]) # 初始化目标Q值列表

        for i in range(BATCH_SIZE): # 目标Q值 BATCH_SIZE=64
            q_rhs = 0
            if (not sample.list_term[i]):
                if self.IsDoubleDQN:  # IsDoubleDQN = False
                    q_rhs=GAMMA * list_pred[i]  # GAMMA =1 
                else:
                    q_rhs=GAMMA * self.Max(list_pred[i]) 
            q_rhs += sample.list_rt[i]  # sample.list_rt[i] 是之前保存在四元组里面的奖励  目标Q值 = 未来奖励+即时奖励
            list_target[i] = q_rhs # 目标Q值更新
            # list_target.append(q_rhs)
        if self.IsPrioritizedSampling:  # IsPrioritizedSampling = False 通过梯度下降更新模型参数，使模型逐渐接近目标 Q 值
            return self.fit_with_prioritized(sample.b_idx,sample.ISWeights,sample.g_list, sample.list_st, sample.list_at,list_target)
        else:
            return self.fit(sample.g_list, sample.list_st, sample.list_at,list_target)

    def fit_with_prioritized(self,tree_idx,ISWeights,g_list,covered,actions,list_target):
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        for i in range(0,n_graphs,BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)
            #node_input = np.array(self.inputs['node_input'])
            #if node_input.size == 0:
            #    continue
            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            result = self.session.run([self.trainStep,self.TD_errors,self.loss],feed_dict={
                                        self.action_select : self.inputs['action_select'],
                                        self.rep_global : self.inputs['rep_global'],
                                        self.n2nsum_param: self.inputs['n2nsum_param'],
                                        self.laplacian_param : self.inputs['laplacian_param'],
                                        self.subgsum_param : self.inputs['subgsum_param'],
                                        self.node_input: np.array(self.inputs['node_input']).reshape(-1, 2),
                                        #self.node_input: self.inputs['node_input'],
                                        self.aux_input : np.array(self.inputs['aux_input']),
                                        self.ISWeights : np.mat(ISWeights).T,  # 多一个这个权重参数
                                        self.target : self.inputs['target']})
            self.nStepReplayMem.batch_update(tree_idx, result[1])
            loss += result[2]*bsize
        # print ('loss')
        # print (loss / len(g_list))
        return loss / len(g_list)


    def fit(self,g_list,covered,actions,list_target):
        cdef double loss = 0.0
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize
        for i in range(0,n_graphs,BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j-i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)

            self.SetupTrain(batch_idxes, g_list, covered, actions,list_target)
            node_input = np.array(self.inputs['node_input'])
            # 执行BuildNet的优化操作，最小化总损失
            result = self.session.run([self.loss, self.trainStep, self.loss_recons],feed_dict={
                                        self.action_select : self.inputs['action_select'],
                                        self.rep_global : self.inputs['rep_global'],
                                        self.n2nsum_param: self.inputs['n2nsum_param'],
                                        self.laplacian_param : self.inputs['laplacian_param'],
                                        self.subgsum_param : self.inputs['subgsum_param'],
                                        #self.node_input: self.inputs['node_input'],
                                        self.node_input: np.array(self.inputs['node_input']).reshape(-1, 2),
                                        self.aux_input: np.array(self.inputs['aux_input']),
                                        self.target : self.inputs['target']})
            loss += result[0]*bsize  #  表示批次所有的损失值
            #print ('Reconstruction los s')
            #print (result[2])
            #print ('loss is')
            #print (loss)

        return loss / len(g_list) 
    #pass

    def Train(self):
        self.PrepareValidData() #准备验证数据
        self.gen_new_graphs(NUM_MIN, NUM_MAX)  # 生成新的图
        cdef int i, iter, idx 
        for i in range(10):
            self.PlayGame(100, 1) #玩100局，探索率为1，获得初步的经验
        self.TakeSnapShot()  # 保存当前模型的快照，保存检查点，避免从头开始每次训练
        cdef double eps_start = 1.0 # 初始探索率
        cdef double eps_end = 0.05 # 最低探索率
        #cdef double eps_step = 10000.0  # 探索率衰减的步数
        cdef double eps_step = 10000.0
        cdef int loss = 0 # 声明损失变量
        cdef double frac, start, end 

        save_dir = './models/Model_%s'%(self.g_type) # 模型保存路径
        print(f"models save_dir  {save_dir}") 
        if not os.path.exists(save_dir): 
            os.mkdir(save_dir)
        # 重新训练要从该文件后面开始写
        VCFile = '%s/ModelVC_%d_%d.csv'%(save_dir, NUM_MIN, NUM_MAX) # 定义验证覆盖率（VC）文件路径
        f_out = open(VCFile, 'a')
        # 这里开始用最新的模型进行训练
        latest_model = './models/Model_barabasi_albert/nrange_30_50_iter_6600.ckpt'
        # 定义重新训练时的迭代次数
        start_iteration = 514500
        try:
            filename = os.path.basename(latest_model)
            iter_str = filename.split('_')[-1].replace('.ckpt', '')
            last_iteration_from_path = int(iter_str)
            print(f"正在加载模型：{latest_model}")
            self.LoadModel(latest_model) # 加载指定的模型
            start_iteration = last_iteration_from_path + 1 # 从上次迭代的下一轮开始
            print(f"将从迭代 {start_iteration} 继续训练。")
        except (ValueError, IndexError) as e:
            print(f"警告：无法从路径 '{latest_model}' 解析出迭代次数，或加载模型失败。")
            print(f"错误信息: {e}")
            print("将从迭代 0 开始训练（从头开始）。")
            start_iteration = 0
            # 如果从头开始，确保初始快照已保存，或根据需要处理
            self.TakeSnapShot() 
        # --- 加载部分结束 ---

        # 重新训练需要修改iter次数
        for iter in range(start_iteration ,MAX_ITERATION):  # 迭代次数，每一次迭代对模型的参数进行更新
            #start = time.clock()
            start = time.perf_counter()
            ###########-----------------------normal training data setup(start) -----------------##############################
            if iter and iter % 5000 == 0:  # 每5000次迭代重新生成图数据
                self.gen_new_graphs(NUM_MIN, NUM_MAX) 
            elif iter == start_iteration and start_iteration % 5000 == 0 and start_iteration > 0:
                # 如果是从一个5000的倍数迭代开始，且不是从0开始，也重新生成一次
                self.gen_new_graphs(NUM_MIN, NUM_MAX)
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step) # 根据当前迭代数动态调整探索率，逐步从eps_start减小到eps_end

            if iter % 10 == 0:  # 每10次迭代
                self.PlayGame(10, eps)  # 玩10局游戏，探索率为pes
            if iter % 300 == 0: # 每300次迭代
                #if(iter == 0):  # 如果是第一次迭代
                #    N_start = start # 记录开始时间
                #else:
                #    N_start = N_end  # 更新时间为上一次的结束时间
                N_start = time.perf_counter()
                frac = 0.0  # 初始化平均覆盖率
                # n_valid = 1
                test_start = time.time()  # 开始测试记录时间
                for idx in range(n_valid): # 遍历验证集
                    frac += self.Test(idx)  # 对每个验证样本进行测试，累加覆盖率(鲁棒性),定期评估，看模型效果好不好
                test_end = time.time()  # 结束测试时间
                f_out.write('%.16f\n'%(frac/n_valid))   #write vc into the file  将验证集平均覆盖率。写入文件
                f_out.flush()  # 刷新文件缓冲区
                print('iter', iter, 'eps', eps, 'average size of vc: ', frac / n_valid)  
                print ('testing 100 graphs time: %.8fs'%(test_end-test_start))
                #N_end = time.clock()
                N_end = time.perf_counter()
                print ('300 iterations total time: %.8fs'%(N_end-N_start))
                sys.stdout.flush()
                model_path = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, iter)
                self.SaveModel(model_path)
            if iter % UPDATE_TIME == 0: # 如果达到模型更新间隔 UPDATE_TIME = 1000
                self.TakeSnapShot()  # 保存模型快照
            self.Fit()  # 模型训练，每次迭代都会调用
        f_out.close()  # 关闭验证覆盖率文件


    def findModel(self):
        VCFile = './models/Model_barabasi_albert1/ModelVC_%d_%d.csv'%(NUM_MIN, NUM_MAX)
        vc_list = []
        for line in open(VCFile):
            vc_list.append(float(line))
        start_loc = 33
        min_vc = start_loc + np.argmin(vc_list[start_loc:])
        best_model_iter = 300 * min_vc
        best_model = './models/Model_barabasi_albert1/nrange_%d_%d_iter_%d.ckpt' % (NUM_MIN, NUM_MAX, best_model_iter)
        return best_model

    def Evaluate1(self, g, save_dir, model_file=None):
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        cdef double frac = 0.0
        cdef double frac_time = 0.0
        result_file = '%s/test.csv' % (save_dir)
        with open(result_file, 'w') as f_out:
            print ('testing')
            sys.stdout.flush()
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol = self.GetSol(0)
            t2 = time.time()
            #for i in range(len(sol)):
            #    f_out.write(' %d\n' % sol[i])
            frac += val
            frac_time += (t2 - t1)
        print ('average size of vc: ', frac)
        print('average time: ', frac_time)

    def Evaluate(self, data_test, model_file):
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        cdef int n_test = 100
        cdef int i
        result_list_score = []
        result_list_time = []
        sys.stdout.flush()
        for i in tqdm(range(n_test)):
            g_path = '%s/'%data_test + 'g_%d'%i
            g = nx.read_gml(g_path)
            btw = {}
            for node in g.nodes():  # This is the crucial fix - iterate through nodes first
                in_weight_sum = sum(
                    float(g[u][node]['weight'])  # 显式转换为float
                    for u in g.predecessors(node)
                )
                # 出边权重和
                out_weight_sum = sum(
                    float(g[node][v]['weight']) 
                    for v in g.successors(node)
                )
                btw[node] = in_weight_sum + out_weight_sum
            # 归一化
            max_btw = max(btw.values()) if btw.values() else 1
            for node in btw:
                btw[node] /= max_btw
            nx.set_node_attributes(g, btw, 'btw')
            if 'in_degree' not in g.nodes[next(iter(g.nodes))]:
                in_degree = dict(g.in_degree())
                out_degree = dict(g.out_degree())
                #btw = nx.betweenness_centrality(g, weight='weight')
                nx.set_node_attributes(g, in_degree, 'in_degree')
                nx.set_node_attributes(g, out_degree, 'out_degree')
                #nx.set_node_attributes(g, btw, 'btw')
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol = self.GetSol(i)
            t2 = time.time()
            result_list_score.append(val)
            result_list_time.append(t2-t1)
        self.ClearTestGraphs()
        # print ('\nvc mean: ', np.mean(result_list))
        # print ('vc std: ', np.std(result_list))
        score_mean = np.mean(result_list_score)
        score_std = np.std(result_list_score)
        time_mean = np.mean(result_list_time)
        time_std = np.std(result_list_time)
        return  score_mean, score_std, time_mean, time_std

    def EvaluateReinsert(self, data_test,save_dir, model_file=None):
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        cdef int n_test = 100
        cdef double frac = 0.0
        cdef double frac_time = 0.0
        cdef int i
        f = open(data_test, 'rb')
        sys.stdout.flush()
        for i in tqdm(range(n_test)):
            g = cp.load(f)
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol = self.GetSolReinsert(i)
            t2 = time.time()
            frac += val
            frac_time += (t2 - t1)
        self.ClearTestGraphs()
        print ('average size of vc: ', frac / n_test)
        print('average time: ', frac_time / n_test)
        return  frac / n_test, frac_time / n_test

    def CleanRealData(self, data_test):
        G = nx.read_weighted_edgelist(data_test)
        cc = sorted(nx.connected_components(G), key = len, reverse=True)
        lcc = cc[0]
        # remove all nodes not in largest component
        numrealnodes = 0
        node_map = {}
        for node in G.nodes():
            if node not in lcc:
                G.remove_node(node)
                continue
            node_map[node] = numrealnodes
            numrealnodes += 1
        # re-create the largest component with nodes indexed from 0 sequentially
        g = nx.Graph()
        for edge in G.edges_iter(data=True):
            src_idx = node_map[edge[0]]
            dst_idx = node_map[edge[1]]
            # w = edge[2]['weight']
            # g.add_edge(src_idx,dst_idx,weight=w)
            g.add_edge(src_idx,dst_idx)
        nx.write_edgelist(g, './Real_data/real1.txt')

    def EvaluateRealData(self, model_file, data_test, save_dir, stepRatio=0.15):  #测试真实数据，原来stepRatio=0.0025
        cdef double solution_time = 0.0
        test_name = data_test.split('/')[-1].replace('.gml','.txt')
        save_dir_local = save_dir+'/StepRatio_%.4f'%stepRatio
        if not os.path.exists(save_dir_local):#make dir
            os.mkdir(save_dir_local)
        result_file = '%s/%s' %(save_dir_local, test_name)
        # g = nx.read_edgelist(data_test)`
        g = nx.read_gml(data_test)
        mapping = {node: int(node) for node in g.nodes if node.isdigit()}  # 确保可转换为整数
        g = nx.relabel_nodes(g, mapping)
        btw = {}
        for node in g.nodes():  # This is the crucial fix - iterate through nodes first
            in_weight_sum = sum(
                float(g[u][node]['weight'])  # 显式转换为float
                for u in g.predecessors(node)
            )
            # 出边权重和
            out_weight_sum = sum(
                float(g[node][v]['weight']) 
                for v in g.successors(node)
            )
            btw[node] = in_weight_sum + out_weight_sum
        # 归一化
        max_btw = max(btw.values()) if btw.values() else 1
        for node in btw:
            btw[node] /= max_btw
        nx.set_node_attributes(g, btw, 'btw')
        in_degree = dict(g.in_degree())
        out_degree = dict(g.out_degree())
        #btw = nx.betweenness_centrality(g, weight='weight')
        nx.set_node_attributes(g, in_degree, 'in_degree')
        nx.set_node_attributes(g, out_degree, 'out_degree')
        with open(result_file, 'w') as f_out:
            print ('testing')
            sys.stdout.flush()
            print ('number of nodes:%d'%(nx.number_of_nodes(g)))
            print ('number of edges:%d'%(nx.number_of_edges(g)))
            if stepRatio > 0:
                step = int(stepRatio*nx.number_of_nodes(g)) #step size
            else:
                step = 1
            self.InsertGraph(g, is_test=True)
            t1 = time.time()
            solution = self.GetSolution(0,step)
            t2 = time.time()
            solution_time = (t2 - t1)
            for i in range(len(solution)):
                #print(solution[i])
                f_out.write('%d\n' % solution[i])
        self.ClearTestGraphs()
        return solution, solution_time

    def EvaRealForModelSelect(self, model_file, data_test, stepRatio=0.0025):  #测试真实数据
        g = nx.read_gml(data_test)
        g_inner = self.GenNetwork(g)
        if stepRatio > 0:
            step = int(stepRatio*nx.number_of_nodes(g)) #step size
        else:
            step = 1
        self.InsertGraph(g, is_test=True)
        sol = self.GetSolution(0,step)
        self.ClearTestGraphs()
        nodes = list(range(nx.number_of_nodes(g)))
        sol_left = list(set(nodes)^set(sol))
        solution = sol + sol_left
        Robustness = self.utils.getRobustness(g_inner, solution)
        return Robustness

    def GetSolution(self, int gid, int step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        num_edges = self.test_env.graph.num_edges
        sol = []
        start = time.time()
        cdef int iter = 0
        cdef int new_action
        while (not self.test_env.isTerminal()):
            print ('Iteration:%d'%iter)
            #print(f"Graph has {num_edges} edges")
            iter += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            #sorted_indices = np.argsort(-list_pred[0])
            #print("sorted_indices:", sorted_indices)
            #print("step:", step)
            #print("batchSol:", batchSol)
            #print("list_pred[0]:", list_pred[0])
            #print("list_pred:", list_pred)
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    break
            #print("*************************8");
        return sol


    def EvaluateSol(self, data_test, sol_file, strategyID=0, reInsertStep=20):
        #evaluate the robust given the solution, strategyID:0,count;2:rank;3:multipy
        sys.stdout.flush()
        # g = nx.read_weighted_edgelist(data_test)
        g = nx.read_gml(data_test)
        btw = {}
        for node in g.nodes():  # This is the crucial fix - iterate through nodes first
            in_weight_sum = sum(
                float(g[u][node]['weight'])  # 显式转换为float
                for u in g.predecessors(node)
            )
            # 出边权重和
            out_weight_sum = sum(
                float(g[node][v]['weight']) 
                for v in g.successors(node)
            )
            btw[node] = in_weight_sum + out_weight_sum
        # 归一化
        max_btw = max(btw.values()) if btw.values() else 1
        for node in btw:
            btw[node] /= max_btw
        nx.set_node_attributes(g, btw, 'btw')
        in_degree = dict(g.in_degree())
        out_degree = dict(g.out_degree())
        #btw = nx.betweenness_centrality(g, weight='weight')
        nx.set_node_attributes(g, in_degree, 'in_degree')
        nx.set_node_attributes(g, out_degree, 'out_degree')
        g_inner = self.GenNetwork(g)
        print ('number of nodes:%d'%nx.number_of_nodes(g))
        print ('number of edges:%d'%nx.number_of_edges(g))
        nodes = list(range(nx.number_of_nodes(g)))
        sol = []
        for line in open(sol_file):
            sol.append(int(line))
        print ('number of sol nodes:%d'%len(sol))
        sol_left = list(set(nodes)^set(sol))
        if strategyID > 0:
            start = time.time()
            sol_reinsert = self.utils.reInsert(g_inner, sol, sol_left, strategyID, reInsertStep)
            end = time.time()
            print ('reInsert time:%.6f'%(end-start))
        else:
            sol_reinsert = sol
        solution = sol_reinsert + sol_left
        print ('number of solution nodes:%d'%len(solution))
        #for node in solution:
        #    print(node)
        Robustness = self.utils.getRobustness(g_inner, solution)
        MaxCCList = self.utils.MaxWccSzList
        CostRatios = self.utils.CostRatios
        CostRatios1 = self.utils.CostRatios1
        return Robustness, MaxCCList, solution, CostRatios,CostRatios1


    def Test(self,int gid):
        g_list = []
        #self.test_env.s0(self.TestSet.Get(gid))
        g_sample = self.TestSet.Get(gid)
        #print(f"g_Sample num_Edges is: {g_sample.num_edges}")
        self.test_env.s0(g_sample)
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0  
        cdef int i
        sol = []  # 存储在测试过程中智能体选择的动作
        while (not self.test_env.isTerminal()):
            # cost += 1
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list]) # 预测下一个动作
            new_action = self.argMax(list_pred[0]) # 选择值最大的动作
            self.test_env.stepWithoutReward(new_action)  # 选择动作，关注策略不是奖励的值,开始一个都没有选择，所以covered为
            sol.append(new_action)
        nodes = list(range(g_list[0].num_nodes)) # 创建一个列表nodes，包含当前图中的所有节点索引
        solution = sol + list(set(nodes)^set(sol)) # 将当前已选择的动作和图中没有被选择的节点合并
        Robustness = self.utils.getRobustness(g_list[0], solution) # 计算鲁棒性 （使用强化学习版）
        return Robustness  # 鲁棒性通常衡量一个策略在面对环境不确定性时的表现


    def GetSol(self, int gid, int step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        #print(f"Number of edges in graph {gid}: {g_list[0].num_edges}")
        cdef double cost = 0.0
        sol = []
        # start = time.time()
        cdef int new_action
        while (not self.test_env.isTerminal()):
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])  # 是因为没有达到终止条件，所以会继续调用此函数，导致出现错误
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    break
        # end = time.time()
        # print ('solution obtained time is:%.8f'%(end-start))
        nodes = list(range(g_list[0].num_nodes))
        solution = sol + list(set(nodes)^set(sol))
        Robustness = self.utils.getRobustness(g_list[0], solution)
        # print ('Robustness is:%.8f'%(Robustness))
        return Robustness, sol

    def GetSolReinsert(self, int gid, int step=1):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        cdef double cost = 0.0
        sol = []
        # start = time.time()
        cdef int new_action
        while (not self.test_env.isTerminal()):
            list_pred = self.PredictWithCurrentQNet(g_list, [self.test_env.action_list])
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    self.test_env.stepWithoutReward(new_action)
                    sol.append(new_action)
                else:
                    continue
        nodes = list(range(g_list[0].num_nodes))
        sol_left = list(set(nodes)^set(sol))
        sol_reinsert = self.utils.reInsert(g_list[0], sol, sol_left, 1, 1)
        solution = sol_reinsert + sol_left
        Robustness = self.utils.getRobustness(g_list[0], solution)
        return Robustness, sol

    def SaveModel(self,model_path):
        self.saver.save(self.session, model_path)
        print('model has been saved success!')

    def LoadModel(self,model_path):
        self.saver.restore(self.session, model_path)
        print('restore model from file successfully')

    def loadGraphList(self, graphFileName):
        with open(graphFileName, 'rb') as f:
            graphList = cp.load(f)
        print("load graph file success!")
        return graphList


    def GenNetwork(self, g):    #networkx2four
        nodes = g.nodes()
        edges = g.edges()
        in_weights = []
        out_weights = []
        Betweenness = []
        edge_weights = []
        for node in nodes:
            in_weights.append(g.nodes[node]['in_degree'])  # 访问节点的入度
            out_weights.append(g.nodes[node]['out_degree'])  # 访问节点的出度
            Betweenness.append(g.nodes[node]['btw']) 
        for u, v in edges:
            edge_weights.append(g[u][v]['weight'])  # 提取边权重

        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
            IW = np.array(in_weights)
            OW = np.array(out_weights)
            BW = np.array(Betweenness)
            EW = np.array(edge_weights)
            
        else:
            A = np.array([0])
            B = np.array([0])
            IW = np.array(in_weights)
            OW = np.array(out_weights)
            BW = np.array(Betweenness)
            EW = np.array(edge_weights)
         # 显式检查所有参数维度
        #print("[DEBUG] 返回 py_Graph 参数数量:", len([len(nodes), len(edges), A, B, IW, OW, BW, EW]))
        #print("[DEBUG] BW.shape:", BW.shape)
        return graph.py_Graph(len(nodes), len(edges), A, B, IW, OW, BW, EW)


    def argMax(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return pos


    def Max(self, scores):
        cdef int n = len(scores)
        cdef int pos = -1
        cdef double best = -10000000
        cdef int i
        for i in range(n):
            if pos == -1 or scores[i] > best:
                pos = i
                best = scores[i]
        return best

    def Isterminal(self, graph):
        if len(nx.edges(graph)) == 0:
            return True
        else:
            return False

    def HXA(self, g, method):
        # 'HDA', 'HBA', 'HPRA', 'HCA'  使用这些度量方法来测试模型的鲁棒性
        sol = []
        G = g.copy()
        in_degree = dict(g.in_degree())
        out_degree = dict(g.out_degree())
        #btw = nx.betweenness_centrality(g, weight='weight')
        nx.set_node_attributes(g, in_degree, 'in_degree')
        nx.set_node_attributes(g, out_degree, 'out_degree')
        #nx.set_node_attributes(g, btw, 'btw')
        while (nx.number_of_edges(G)>0):
            if method == 'HDA':  # 度中心性：衡量节点与其它节点的直接连接数
                dc = nx.in_degree_centrality(G)
            elif method == 'HBA': # 介数中心性 衡量节点作为其他节点间最短路径中间节点的重要性
                dc = nx.betweenness_centrality(G)
            elif method == 'HCA': # 接近中心性 衡量节点与图中其他节点的平均距离
                dc = nx.closeness_centrality(G)
            elif method == 'HPRA':  # 衡量节点在图中重要性
                dc = nx.pagerank(G)
            keys = list(dc.keys()) # 提取所有节点
            values = list(dc.values()) # 提取所借节点的中心性值
            maxTag = np.argmax(values) # 找到中心性值最大的索引
            node = keys[maxTag] # 找到当前中心性值最大的节点
            sol.append(node) # 将节点添加到解中
            G.remove_node(node)
        solution = sol + list(set(g.nodes())^set(sol))
        solutions = [int(i) for i in solution]
        #print("Solutions size:", len(solutions))  # 打印 solutions 的大小
        Robustness = self.utils.getRobustness(self.GenNetwork(g), solutions)
        #print(f"Removed node {node}. Current number of nodes: {G.number_of_nodes()}, Current number of edges: {G.number_of_edges()}")
        return Robustness

    def HXA1(self, g, method):
        # 'HDA', 'HBA', 'HPRA', 'HCA'  使用这些度量方法来测试模型的鲁棒性
        sol = []
        G = g.copy()
        in_degree = dict(g.in_degree())
        out_degree = dict(g.out_degree())
        #btw = nx.betweenness_centrality(g, weight='weight')
        nx.set_node_attributes(g, in_degree, 'in_degree')
        nx.set_node_attributes(g, out_degree, 'out_degree')
        #nx.set_node_attributes(g, btw, 'btw')
        while (nx.number_of_edges(G)>0):
            if method == 'HDA':  # 度中心性：衡量节点与其它节点的直接连接数
                dc = nx.in_degree_centrality(G)
            elif method == 'HBA': # 介数中心性 衡量节点作为其他节点间最短路径中间节点的重要性
                dc = nx.betweenness_centrality(G)
            elif method == 'HCA': # 接近中心性 衡量节点与图中其他节点的平均距离
                dc = nx.closeness_centrality(G)
            elif method == 'HPRA':  # 衡量节点在图中重要性
                dc = nx.pagerank(G)
            keys = list(dc.keys()) # 提取所有节点
            values = list(dc.values()) # 提取所借节点的中心性值
            maxTag = np.argmax(values) # 找到中心性值最大的索引
            node = keys[maxTag] # 找到当前中心性值最大的节点
            sol.append(node) # 将节点添加到解中
            G.remove_node(node)
        solution = sol + list(set(g.nodes())^set(sol))
        solutions = [int(i) for i in solution]
        #print("Solutions size:", len(solutions))  # 打印 solutions 的大小
        Robustness = self.utils.getRobustness(self.GenNetwork(g), solutions)
        #print(f"Removed node {node}. Current number of nodes: {G.number_of_nodes()}, Current number of edges: {G.number_of_edges()}")
        MaxCCList = self.utils.MaxWccSzList
        Cost = self.utils.CostRatios
        Cost1 = self.utils.CostRatios1
        return Robustness,MaxCCList,Cost,Cost1

    def Real2networkx(self, G):
        cc = sorted(nx.connected_components(G), key=len, reverse=True)
        lcc = cc[0]
        # remove all nodes not in largest component
        numrealnodes = 0
        node_map = {}
        G_copy = G.copy()
        nodes = G_copy.nodes()
        weights = {}
        for node in nodes:
            if node not in lcc:
                G.remove_node(node)
                continue
            node_map[node] = numrealnodes
            weights[numrealnodes] = G_copy.node[node]['weight']
            numrealnodes += 1
        # re-create the largest component with nodes indexed from 0 sequentially
        g = nx.Graph()
        for edge in G.edges():
            src_idx = node_map[edge[0]]
            dst_idx = node_map[edge[1]]
            g.add_edge(src_idx, dst_idx)
        nx.set_node_attributes(g, 'weight', weights)
        return g

    def Evaluate4Vis(self, g, model_file=None):
        if model_file == None:  #if user do not specify the model_file
            model_file = self.findModel()
        print ('The best model is :%s'%(model_file))
        sys.stdout.flush()
        self.LoadModel(model_file)
        print ('testing')
        sys.stdout.flush()
        self.InsertGraph(g, is_test=True)
        sol = self.GetSol4Vis(0)
        print (sol)
        fout_sol = open('./Visualize/PKL/Sol.pkl','ab')
        cp.dump(sol, fout_sol)
        fout_sol.close()


    def GetSol4Vis(self,int gid):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        sol = []
        cdef int iter = 0
        while (not self.test_env.isTerminal()):
            fout_q = open('./Visualize/PKL/Q_%d.pkl'%iter,'ab')
            fout_embed = open('./Visualize/PKL/Embed_%d.pkl'%iter,'ab')
            list_pred, list_embed = self.Predict4Vis(g_list, [self.test_env.action_list])
            cp.dump(list_pred[0], fout_q)
            cp.dump(list_embed[0], fout_embed)
            fout_q.close()
            fout_embed.close()
            new_action = self.argMax(list_pred[0])
            self.test_env.stepWithoutReward(new_action)
            sol.append(new_action)
            iter += 1
        return sol


    def Predict4Vis(self,g_list,covered):
        cdef int n_graphs = len(g_list)
        cdef int i, j, k, bsize
        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)

            idx_map_list = self.SetupPredAll(batch_idxes, g_list, covered)
            q_all, node_embed = self.session.run([self.q_on_all, self.node_embedding], feed_dict={
                self.rep_global: self.inputs['rep_global'],
                self.n2nsum_param: self.inputs['n2nsum_param'],
                self.subgsum_param: self.inputs['subgsum_param'],
                #self.node_input: self.inputs['node_input'],
                self.node_input: np.array(self.inputs['node_input']).reshape(-1, 2),
                self.aux_input: np.array(self.inputs['aux_input'])})

            raw_output = q_all
            raw_embed = node_embed
            # print("#####@!@@@")
            # print(np.shape(node_embed[0]))
            pos = 0
            pred = []
            embed = []
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j-i]
                cur_pred = np.zeros(len(idx_map))
                cur_embed = []
                for m in range(len(idx_map)):
                    cur_embed.append([])
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf
                        cur_embed[k] = []
                    else:
                        cur_pred[k] = raw_output[pos]
                        cur_embed[k] = raw_embed[pos]
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -inf
                    cur_embed[k] = []
                pred.append(cur_pred)
                embed.append(cur_embed)
            assert (pos == len(raw_output))
        return pred,embed

        