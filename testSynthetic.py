#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from DWKEY import DWKEY
from tqdm import tqdm



def main():
    dqn = DWKEY()
    cost_types = ['sw_cost']
    for cost in cost_types:
        data_test_path = '/home/renhuiling/Code_Ren/data/synthetic/%s/'%cost
        data_test_name = ['30-50','50-100','100-200','200-300','300-400','400-500']
        model_file = './models/Model_barabasi_albert/nrange_30_50_iter_298800.ckpt'
        
        file_path = '../results/DWKEY_ND_cost/sw'
        if not os.path.exists('../results/DWKEY_ND_cost'):
            os.mkdir('../results/DWKEY_ND_cost')
        if not os.path.exists('../results/DWKEY_ND_cost/sw'):
            os.mkdir('../results/DWKEY_ND_cost/sw')
        
        with open('%s/%s_score.txt'%(file_path, cost), 'w') as fout:
            for i in tqdm(range(len(data_test_name))):
                data_test = data_test_path + data_test_name[i]
                score_mean, score_std, time_mean, time_std = dqn.Evaluate(data_test, None)
                fout.write('%.2f±%.2f,' % (score_mean * 100, score_std * 100))
                fout.flush()
                print('data_test_%s has been tested!' % data_test_name[i])

if __name__=="__main__":
    main()
