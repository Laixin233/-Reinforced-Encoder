from task.TaskLoader import Opt, TaskDataset
import numpy as np
import os
import pandas as pd

from data.base import nn_base

from ray import tune


class ili_data(TaskDataset):
    def __init__(self, args):
        super().__init__(args)
    
    def info_config(self):
        self.info.normal = False
        self.info.data_path = 'data/paper/seq/real/illness/national_illness.csv' #'data/paper/seq/real/ili/ILI.csv'
        # self.info.series_name = ['south','north']
        # self.info.num_series = 2
        self.info.series_name = ['national_illness']
        self.info.num_series = 1
        self.info.cov_dim = 0
        self.info.steps = 48 #96
        self.info.H = 12

        self.info.batch_size = 16
        self.info.scale = True
        self.info.if_patch = False
        self.info.feature = 'MS'
        self.info.patch_size = 6#6
        self.info.input_dim = 1
        self.info.period = 4
        self.info.stride = 6
        self.info.num_variate = 7

    def sub_config(self,):
        self.seriesPack = []
            
        hostDir_path = 'data/paper/esn/ili/'
        for i, name in enumerate(self.info.series_name):
            # npy_file = os.path.join(hostDir_path, '{}.npy'.format(name))
            # if os.path.exists(npy_file):
            #     raw_ts = np.load(npy_file)
            # else:
            df = pd.read_csv(self.info.data_path, header=0)

            if self.info.num_variate == 1:
                raw_ts = df.values[:,-1]#_start
            else:
                raw_ts = df.values[:,1:]
                    
            sub = self.loader_dataset(raw_ts)
            # sub = self.pack_dataset(raw_ts) #dataloader
            sub.index = i
            sub.name = name
            sub.H = self.info.H
            sub.merge(self.info)            

            self.seriesPack.append(sub)


class seq2seq_base(nn_base):
    def __init__(self):
        self.training = False
        self.arch = 'seq2seq'
        self.model_path = 'models/RLmodel'
        super().__init__()
        
    def hyper_init(self,):
        self.hyper.epochs = 20 # 100
        self.hyper.preTrain_epochs = 20
        self.hyper.base_iter = 10
        self.hyper.ppo_epochs = 10
        self.hyper.agent_iter = 30 #50
        self.hyper.env_iter = 20 #20
        
        self.hyper.component = 'LSTM' #['RNN','MGU','GRU','LSTM', 'IndRNN', 'PeepholeLSTM','PhaseLSTM','xLSTM']
        self.hyper.optimizer = 'adam'#rmsprop:xlstm,adam:peepholeLSTM
        # tune
        self.hyper.encoder_hidden_size = 60 # 65:xlstm; 30:rnn,IndRNN，phasedlstm； 120：MGU
        self.hyper.encoder_num_layer = 2 # 2 # xlstm：2

        self.hyper.learning_rate = 0.01 # xlstm：0.05 
        self.hyper.step_gamma = 0.92 # xlstm：0.91 GRU:0.82 
        
        self.hyper.agent_hidden_size = 128 #256
        self.hyper.agent_num_layer = 2
        self.hyper.agent_nhead = 4
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.agent_lr = 0.01        
        self.hyper.agent_step_gamma = 0.9
        self.hyper.error_rate = 0.1        
        self.hyper.agent_model = 'Transformer' #['MLP','RNN','Transformer']
    
        self.hyper.error_rate = 0.1

        self.hyper.patch_size = 12
        self.hyper.T = 48
        self.hyper.H = 12
        self.hyper.num_variate = 7
        self.hyper.final_gamma = 10 # 最后一步的奖励系数,>1
        self.hyper.jump_step = 4
    
    def tuning_init(self,):
        # self.hyper.epochs = 50   # fitness epoch per iter
        # self.tuner.iters = 30  # tuner search times
        # self.tuner.points_eval = None  # points_to_evaluate or not
        # self.tuner.name = 'Random_Search' #['Random_Search','Hyperopt_Search','Ax_Search']
        
        # total cpu cores for tuning
        self.tuner.cores = 12
        # gpu cards per trial in tune
        self.tuner.cards = 2
        # fitness repeat_num per iter
        self.tuner.epochPerIter = 1

class RNNPPO(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        # self.import_path = 'models/RLmodel/Multi_batch_single_variate/PPO_patch.py' #
        self.model='PPO4Pred.py' #PPO_patch,PPO_ER
        self.import_path = os.path.join(self.model_path, self.model)
        self.class_name = 'rnnPPO'
        #self.import_path = 'models/seq2seq/RNN_patch_randn.py'
        #self.class_name = 'randnRNN'
    
    def hyper_modify(self):  
        # self.hyper.base_iter = 10
        # self.hyper.preTrain_epochs = 20  
        self.hyper.ppo_epochs = 5
        
        self.hyper.output_agent = False
        self.hyper.num_actions=3
        
        self.hyper.reward_alpha = 1.0  # 奖励敏感系数
        self.hyper.reward_threshold = 0.55  # 阈值 c
        self.hyper.priority_beta = 0.5  # TD误差和预测误差的权重 [0,1]
        self.hyper.lambda_min = 0.1  # 最小温度
        self.hyper.lambda_max = 2.0  # 最大温度
        self.hyper.temp_oscillation = 0.1  # 温度振荡幅度
        self.hyper.temp_frequency = 2.0  # 温度周期频率

        # env
        # self.hyper.encoder_hidden_size = 120 # 65:xlstm; 30:rnn IndRNN，phasedlstm:120 , MGU GRU:120 LSTM：120 
        # self.hyper.encoder_num_layer = 2 # 2 # xlstm：2

        self.hyper.learning_rate = 0.006 # xlstm：0.05, RNN：0.001, MGU：0.005, GRU：0.01, LSTM:0.008, phasedlstm：0.006
        self.hyper.step_gamma = 0.86 # RNN，xlstm：0.91,  MGU：0.9, GRU:0.82，LSTM:0.85, phasedlstm：0.86

        # H = 12 
        # tune H = 12  根据RL寻优得到

        self.hyper.agent_hidden_size = 128 #256
        self.hyper.agent_dropout_rate = 0.3
        self.hyper.agent_lr = 0.001
        self.hyper.agent_step_gamma = 0.8
        self.hyper.error_rate = 0.1

        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  =  0.005
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.38
        self.hyper.clip_param = 0.2
        self.hyper.max_grad_norm = 0.5

        self.hyper.buffer_capacity = 100
        self.hyper.gamma = 0.9

        self.hyper.data_store_maxsize = 1000
        self.hyper.T_step = 4000
        # self.hyper.patch_size = 6

        self.hyper.dropout_rate = 0.3
        self.hyper.encoder_num_layer = 2
        
    def tuning_modify(self):
        self.tuning.error_rate = tune.quniform(0,1,0.1)
        self.tuning.agent_lr = tune.loguniform(1e-5, 1e-1)
        # self.tuning.agent_lr = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
        self.tuning.agent_step_gamma = tune.quniform(0.90,0.99,0.01)
        self.tuning.agent_hidden_size = tune.qrandint(32,512, 32)
        self.tuning.agent_dropout_rate = tune.quniform(0.1,0.9,0.1)



