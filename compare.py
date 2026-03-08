#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from DWKEY import DWKEY
import numpy as np
import pandas as pd
import time
import json
import networkx as nx

def RunHXAComparison(data_path, dataset, cost_type, output_dir):
    """运行HXA不同方法的对比实验"""
    # 初始化DWKEY
    dqn = DWKEY()
    
    # 加载数据
    graph_file = f"{data_path}{dataset}_{cost_type}.gml"
    g = nx.read_gml(graph_file)
    # 要测试的方法列表
    methods = ['HDA', 'HBA', 'HCA', 'HPRA']
    
    # 存储结果
    results = {}
    
    for method in methods:
        print(f"Running {method} on {dataset}...")
        start_time = time.time()
        
        # 调用HXA方法
        robustness,MaxCCList,Cost,Cost1 = dqn.HXA(g, method)
        print(f"Method: {method}, Robustness: {robustness}")  # 打印robustness值
        
        # 记录结果
        results[method] = {
            'robustness': robustness,
            'time': time.time() - start_time
        }
    
    # 为每个方法单独保存MaxCCList到txt文件
        os.makedirs(output_dir, exist_ok=True)
        list_file = f"{output_dir}/{dataset}_{method}_MaxCCList.txt"
        np.savetxt(list_file, MaxCCList, fmt='%.8f')
        list_file_1 = f"{output_dir}/{dataset}_{method}_Cost.txt"
        np.savetxt(list_file_1, Cost, fmt='%.8f')
        list_file_2 = f"{output_dir}/{dataset}_{method}_Cost1.txt"
        np.savetxt(list_file_2, Cost1, fmt='%.8f')
        print(f"MaxCCList saved to {list_file}")
        print(f"Cost saved to {list_file_1}")
        print(f"Cost1 saved to {list_file_2}")

    
    # 保存CSV结果
    df_data = []
    for method, res in results.items():
        df_data.append({
            'method': method,
            'robustness': res['robustness'],
            'time': res['time']
        })
    
    # 同时保存为CSV便于查看
    df = pd.DataFrame.from_dict(results, orient='index')
    csv_file = f"{output_dir}/{dataset}_HXA_comparison.csv"
    df.to_csv(csv_file)
    print(f"CSV results saved to {csv_file}")

def main():
    # 配置参数
    data_path = '/home/renhuiling/Code_Ren/data/real/Cost/'
    datasets = ['spanish']  # 可以添加更多数据集
    cost_types = ['degree']  # 也可以测试random
    output_dir = '../results/HXA_Comparison'
    
    # 运行实验
    for dataset in datasets:
        for cost_type in cost_types:
            RunHXAComparison(data_path, dataset, cost_type, output_dir)

if __name__ == "__main__":
    main()