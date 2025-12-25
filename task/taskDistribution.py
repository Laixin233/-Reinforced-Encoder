import os
import sys
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import shutil
from tqdm import trange
import statistics
import numpy as np
import torch
from task.TaskLoader import Opt
from task.TaskTuner import StocHyperTuner as HyperTuner
from task.dataset import de_scale
from task.util import os_makedirs, os_rmdirs, set_logger
from task.util import plot_xfit,plot_xfit_agent,plot_hError,scaler_inverse
from models.statistical.naive import Naive
import importlib
from tqdm.std import tqdm
import math
import matplotlib.pyplot as plt
from scipy import stats as st


class Task_Distri(Opt):
    def __init__(self, args):
        # self.opts = Opt()
        # self.opts.merge(args)

        self.exp_module_path = importlib.import_module('data.{}.config.{}'.format(
            args.datafolder.replace('/', '.'), args.dataset))  # 引入配置
        self.data_config(args)
        self.data_subconfig()

    def data_config(self, args):
        self.data_name = args.dataset
        data_opts = getattr(self.exp_module_path, args.dataset + '_data')
        self.data_opts = data_opts(args)

    def data_subconfig(self,):
        self.data_opts.arch = 'mlp'
        self.data_opts.sub_config()

    def conduct(self,):
        for sub_count, series_Pack in enumerate(self.data_opts.seriesPack):
            assert sub_count == series_Pack.index
            train_data = series_Pack.train_loader
            valid_data = series_Pack.valid_loader
            test_data = series_Pack.test_loader
            labels = ['train','valid','test']
            # 当相邻的两步相差较大时，expoure bias将表现的比较明显
            data_ = [train_data,valid_data,test_data]
            for i in range(len(data_)):
                x = []
                y = []
                for batch_x, batch_y in data_[i]:
                    x.append(batch_x)
                    y.append(batch_y)
                x = torch.cat(x, dim=0)
                y = torch.cat(y, dim=0)
                # y_t = torch.cat([x[:,-1].unsqueeze(1),y[:,:-1]],dim=1).detach().cpu().numpy()
                # y_t_plus = y.detach().cpu().numpy()
                # deta_y = np.mean(y_t_plus - y_t,axis=0)
                y_t = x[:,-1].detach().cpu().numpy()
                y_t_plus = y[:,0].detach().cpu().numpy()
                deta_y = y_t_plus - y_t
                # self.test_(y_0,y_all,labels[i])
                print('dataset:{}\t  H:{}\t Label:{} \t (y_t+1) - (y_t): 均值{}\t 标准差{}'.format(self.data_opts.args.dataset,self.data_opts.args.H,labels[i],np.mean(deta_y),np.std(deta_y)))
                self.plot_(deta_y,None,labels[i],_legend=["y_(t+1)-y_(t)",""])
    def test_(self,data_1,data_2,label):
        #均值检验结果
        meantest=[]
        np.array(meantest)
        #中位数检验结果
        mediantest=[]
        np.array(mediantest)
        #检验
        # cols=labels
        if st.levene(data_1,data_2).pvalue > 0.05 :
            # 样本具备方差齐性
            var_H = True
        else:
            var_H = False
        if st.ttest_ind(data_1,data_2,equal_var=var_H).pvalue < 0.05:
            print('dataset:{}\t  H:{}\t  Label:{}\t  具有显著差异p_value{}'.format(self.data_opts.args.dataset,self.data_opts.args.H,label,st.ttest_ind(data_1,data_2,equal_var=var_H).pvalue))
        else:
            print('dataset:{}\t  H:{}\t  Label:{}\t  不具有显著差异p_value{}'.format(self.data_opts.args.dataset,self.data_opts.args.H,label,st.ttest_ind(data_1,data_2,equal_var=var_H).pvalue))
    def plot_(self,data1,data2,label,_legend):
        x_ =  list(range(1,len(data1)+1))
        plt.figure()
        if data1 is not None:
            plt.plot(x_,data1,label=_legend[0])
        if data2 is not None:    
            plt.plot(x_,data2,label = _legend[1])
        plt.legend()
        location = '_ResultAnalysis/Distribution/{}/H{}'.format(self.data_opts.args.dataset,self.data_opts.args.H)
        if os.path.exists(location) is False:
            os_makedirs(location)
        plt.savefig(location +'/{}.png'.format(label))
        
                
                
                
            