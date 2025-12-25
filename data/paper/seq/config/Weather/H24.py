from ray import tune
from data.base import nn_base

from task.TaskLoader import TaskDataset, Opt

import numpy as np
import pandas as pd
import os

class Weather_data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)
        
    def info_config(self):
        self.info.normal = False
        self.info.data_path = 'data/paper/seq/real/weather/'
        
        # self.info.series_name = ['ETTh1','ETTh2']  
        
        self.info.series_name = ['weather']
        self.info.num_series = len(self.info.series_name) 
        self.start_time = '2020-05-20 00:00:00'
        
        self.info.cov_dim = 0
        # self.info.period = 24
        self.info.batch_size = 256
        # self.info.num_variate = 7
        self.info.steps = 96
        self.info.H = 24

        self.info.scale = True
        self.info.if_patch = False
        self.info.feature = 'MS'
        self.info.patch_size = 12 #6，12
        self.info.stride = 6
        self.info.period = 4
        self.info.num_variate = 21 # 21
        

    def sub_config(self,):
        self.seriesPack = []
        
        for i in range(self.info.num_series):
            _name = self.info.series_name[i]
            df = pd.read_csv(self.info.data_path + self.info.series_name[i] + '.csv',index_col=None,header=0)
            _start = np.where(df.values==self.start_time)[0].item()
            if self.info.num_variate == 1:
                raw_ts = df.values[:,-1] #跑全部数据去掉_start
            else:
                raw_ts = df.values[:,1:]#去掉时间

            sub = self.loader_dataset(raw_ts)
            # sub = self.pack_dataset(raw_ts) #dataloader
            sub.index = i
            sub.name = _name
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
        self.hyper.optimizer = 'adam'#RMSprop：xlstm,adam:peepholeLSTM
        # tune
        self.hyper.encoder_hidden_size = 100 # xlstm：65;rnn,IndRNN，phasedlstm:30
        self.hyper.encoder_num_layer = 2 # 2 # xlstm：2
        self.hyper.decoder_hidden_size = 200
        self.hyper.decoder_num_layer = 2
        self.hyper.decoder_input_size = 85
        self.hyper.learning_rate = 0.01 # xlstm：0.05
        self.hyper.step_gamma = 0.90 # xlstm：0.91
        self.hyper.final_gamma = 10 # 最后一步的奖励系数,>1

        
        self.hyper.agent_hidden_size = 128 #256
        self.hyper.agent_num_layer = 2
        self.hyper.agent_nhead = 4
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.agent_lr = 0.01        
        self.hyper.agent_step_gamma = 0.9
        self.hyper.error_rate = 0.1        
        self.hyper.agent_model = 'Transformer' #['MLP','RNN','Transformer']


        self.hyper.patch_size = 12
        self.hyper.T = 96
        self.hyper.H = 24
        self.hyper.final_gamma = 10 # 最后一步的奖励系数,>1
        self.hyper.jump_step = 4

    def tuning_init(self,):
        # self.hyper.epochs = 50   # fitness epoch per iter
        # self.tuner.iters = 30  # tuner search times
        self.tuner.points_eval = None  # points_to_evaluate or not
        self.tuner.name = 'Random_Search' #['Random_Search','Hyperopt_Search','Ax_Search']
        
        # total cpu cores for tuning
        self.tuner.cores = 12
        # gpu cards per trial in tune
        self.tuner.cards = 2
        # fitness repeat_num per iter
        self.tuner.epochPerIter = 1


class RNNPG(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        # self.import_path = 'models/RLmodel/Multi_batch_single_variate/PG_patch.py'
        self.model='PG_patch.py'
        self.import_path = os.path.join(self.model_path, self.model)
        self.class_name = 'rnnPG'
    
    def hyper_modify(self):  
        self.hyper.base_iter = 10
        self.hyper.preTrain_epochs = 10  
        
        self.hyper.output_agent = False

        self.hyper.optimizer = 'adam'#RMSprop：xlstm,adam:peepholeLSTM
        self.hyper.encoder_hidden_size = 400 # 400
        self.hyper.encoder_num_layer = 3 # 2  
        self.hyper.learning_rate = 0.01 # 0.01
        self.hyper.step_gamma = 0.91 # 0.91

        self.hyper.agent_hidden_size_1 = 1000
        self.hyper.agent_hidden_size = 500 # 256
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.agent_lr = 0.01 # 0.001
        self.hyper.agent_step_gamma = 0.9
        self.hyper.error_rate = 0.1

        self.hyper.mlp_hidden_size = 500
        self.hyper.mlp_lr  =  0.005
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.38

        
    def tuning_modify(self):
        self.tuning.error_rate = tune.quniform(0,1,0.1)
        self.tuning.agent_lr = tune.loguniform(1e-5, 1e-1)
        # self.tuning.agent_lr = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
        self.tuning.agent_step_gamma = tune.quniform(0.90,0.99,0.01)
        self.tuning.agent_hidden_size = tune.qrandint(100,1000, 100)
        self.tuning.agent_dropout_rate = tune.quniform(0.1,0.9,0.1)
        self.tuning.encoder_hidden_size = tune.qrandint(100,1000, 100) 
        self.tuning.step_gamma = tune.quniform(0.90,0.99,0.01)
        self.tuning.learning_rate = tune.loguniform(1e-5, 1e-1)


class DilatedRNN(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):

        self.model='Dilated-RNN.py'
        self.import_path = os.path.join(self.model_path, self.model)
        self.class_name = 'Dilatedrnn'

    
    def hyper_modify(self):  
        self.hyper.base_iter = 10
        self.hyper.preTrain_epochs = 10
        
        self.hyper.output_agent = False
        self.hyper.num_actions=1
        self.hyper.fixed_window = 3

        # H = 12 
        # tune H = 12  根据RL寻优得到
        self.hyper.agent_hidden_size_1 = 1000
        self.hyper.agent_hidden_size = 500 #256
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.agent_lr = 0.001
        self.hyper.agent_step_gamma = 0.8
        self.hyper.error_rate = 0.1
        # # tune H = 12  根据Type2
        # self.hyper.agent_hidden_size = 352
        # self.hyper.agent_dropout_rate = 0.7
        # self.hyper.agent_lr = 0.05
        # self.hyper.agent_step_gamma = 0.98
        # self.hyper.error_rate = 1
        # # H = 12 根据MLP寻优结果确认
        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  =  0.005
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.38
        self.hyper.clip_param = 0.2
        self.hyper.max_grad_norm = 0.5
        
        self.hyper.buffer_capacity = 100
        self.hyper.gamma = 0.9
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


class LSTMJUMP(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):

        self.model='LSTM-jump.py'
        self.import_path = os.path.join(self.model_path, self.model)
        self.class_name = 'Lstmjump'

    
    def hyper_modify(self):  
        self.hyper.base_iter = 10
        # self.hyper.preTrain_epochs = 10  
        
        self.hyper.output_agent = False
        self.hyper.num_actions=3

        # H = 12 
        # tune H = 12  根据RL寻优得到
        self.hyper.agent_hidden_size_1 = 1000
        self.hyper.agent_hidden_size = 500 #256
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.agent_lr = 0.001
        self.hyper.agent_step_gamma = 0.8
        self.hyper.error_rate = 0.1
        # # tune H = 12  根据Type2
        # self.hyper.agent_hidden_size = 352
        # self.hyper.agent_dropout_rate = 0.7
        # self.hyper.agent_lr = 0.05
        # self.hyper.agent_step_gamma = 0.98
        # self.hyper.error_rate = 1
        # # H = 12 根据MLP寻优结果确认
        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  =  0.005
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.38
        self.hyper.clip_param = 0.2
        self.hyper.max_grad_norm = 0.5
        
        self.hyper.buffer_capacity = 100
        self.hyper.gamma = 0.9
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



class PGLSTM(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):

        self.model='PG-LSTM.py'
        self.import_path = os.path.join(self.model_path, self.model)
        self.class_name = 'Pglstm'

    
    def hyper_modify(self):  
        self.hyper.base_iter = 10
        # self.hyper.preTrain_epochs = 10  
        
        self.hyper.output_agent = False
        self.hyper.num_actions=3

        # H = 12 
        # tune H = 12  根据RL寻优得到
        self.hyper.agent_hidden_size_1 = 1000
        self.hyper.agent_hidden_size = 500 #256
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.agent_lr = 0.001
        self.hyper.agent_step_gamma = 0.8
        self.hyper.error_rate = 0.1
        # # tune H = 12  根据Type2
        # self.hyper.agent_hidden_size = 352
        # self.hyper.agent_dropout_rate = 0.7
        # self.hyper.agent_lr = 0.05
        # self.hyper.agent_step_gamma = 0.98
        # self.hyper.error_rate = 1
        # # H = 12 根据MLP寻优结果确认
        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  =  0.005
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.38
        self.hyper.clip_param = 0.2
        self.hyper.max_grad_norm = 0.5
        
        self.hyper.buffer_capacity = 100
        self.hyper.gamma = 0.9
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

class EBPSO(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):

        self.model='RNN_EBPSO.py'
        self.import_path = os.path.join(self.model_path, self.model)
        self.class_name = 'Ebpso'

    
    def hyper_modify(self):  
        self.hyper.base_iter = 1
        self.hyper.preTrain_epochs = 10  
        
        self.hyper.output_agent = False
        self.hyper.num_actions=3
        self.hyper.n_particles = 20
        self.hyper.alpha = 0.99
        self.hyper.beta = 0.01
        self.hyper.pca_components = 0.95
        self.hyper.k_EBPSO = 5



        # H = 12 
        # tune H = 12  根据RL寻优得到
        self.hyper.agent_hidden_size_1 = 1000
        self.hyper.agent_hidden_size = 500 #256
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.agent_lr = 0.001
        self.hyper.agent_step_gamma = 0.8
        self.hyper.error_rate = 0.1
        # # tune H = 12  根据Type2
        # self.hyper.agent_hidden_size = 352
        # self.hyper.agent_dropout_rate = 0.7
        # self.hyper.agent_lr = 0.05
        # self.hyper.agent_step_gamma = 0.98
        # self.hyper.error_rate = 1
        # # H = 12 根据MLP寻优结果确认
        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  =  0.005
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.38
        self.hyper.clip_param = 0.2
        self.hyper.max_grad_norm = 0.5
        
        self.hyper.buffer_capacity = 100
        self.hyper.gamma = 0.9
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



class RNNPG(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        # self.import_path = 'models/RLmodel/Multi_batch_single_variate/PG_patch.py'
        self.model='PG_patch.py'
        self.import_path = os.path.join(self.model_path, self.model)
        self.class_name = 'rnnPG'
    
    def hyper_modify(self):  
        self.hyper.base_iter = 10
        self.hyper.preTrain_epochs = 10  
        
        self.hyper.output_agent = False

        self.hyper.optimizer = 'adam'#RMSprop：xlstm,adam:peepholeLSTM
        self.hyper.encoder_hidden_size = 400 # 400
        self.hyper.encoder_num_layer = 3 # 2  
        self.hyper.learning_rate = 0.01 # 0.01
        self.hyper.step_gamma = 0.91 # 0.91

        self.hyper.agent_hidden_size_1 = 1000
        self.hyper.agent_hidden_size = 500 # 256
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.agent_lr = 0.01 # 0.001
        self.hyper.agent_step_gamma = 0.9
        self.hyper.error_rate = 0.1

        self.hyper.mlp_hidden_size = 500
        self.hyper.mlp_lr  =  0.005
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.38

        
    def tuning_modify(self):
        self.tuning.error_rate = tune.quniform(0,1,0.1)
        self.tuning.agent_lr = tune.loguniform(1e-5, 1e-1)
        # self.tuning.agent_lr = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
        self.tuning.agent_step_gamma = tune.quniform(0.90,0.99,0.01)
        self.tuning.agent_hidden_size = tune.qrandint(100,1000, 100)
        self.tuning.agent_dropout_rate = tune.quniform(0.1,0.9,0.1)
        self.tuning.encoder_hidden_size = tune.qrandint(100,1000, 100) 
        self.tuning.step_gamma = tune.quniform(0.90,0.99,0.01)
        self.tuning.learning_rate = tune.loguniform(1e-5, 1e-1)

class RNNDQN(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        # self.import_path = 'models/RLmodel/Multi_batch_single_variate/DQN_patch.py'
        self.model='DQN_patch.py'
        self.import_path = os.path.join(self.model_path, self.model)
        self.class_name = 'rnnDQN'
    
    def hyper_modify(self):  
        self.hyper.base_iter = 10
        self.hyper.preTrain_epochs = 10  
        
        self.hyper.output_agent = False

        # self.hyper.patch_size = 6

        # H = 12 
        # tune H = 12  根据RL寻优得到
        self.hyper.agent_hidden_size_1 = 1000
        self.hyper.agent_hidden_size = 256 # phasedlstm：256 xlstm：500 
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.agent_lr = 0.001 #0.04227805743904207
        self.hyper.agent_step_gamma = 0.8
        self.hyper.error_rate = 0.1
        # # tune H = 12  根据Type2
        # self.hyper.agent_hidden_size = 352
        # self.hyper.agent_dropout_rate = 0.7
        # self.hyper.agent_lr = 0.05
        # self.hyper.agent_step_gamma = 0.98
        # self.hyper.error_rate = 1
        # # H = 12 根据MLP寻优结果确认
        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  =  0.005
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.38
        self.hyper.num_actions = 3
        self.hyper.dropout_rate = 0.3
        self.hyper.data_store_maxsize = 1000
        self.hyper.gamma = 0.6 #0.9
        self.hyper.memory_size = 1000
        self.hyper.target_replace_iter = 100
        
    def tuning_modify(self):
        self.tuning.error_rate = tune.quniform(0,1,0.1)
        self.tuning.agent_lr = tune.loguniform(1e-5, 1e-1)
        # self.tuning.agent_lr = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
        self.tuning.agent_step_gamma = tune.quniform(0.90,0.99,0.01)
        self.tuning.agent_hidden_size = tune.qrandint(32,512, 32)
        self.tuning.agent_dropout_rate = tune.quniform(0.1,0.9,0.1)

class RNNPPO(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        
        self.model='PPO4Pred.py'
        self.import_path = os.path.join(self.model_path, self.model)
        self.class_name = 'rnnPPO'

    
    def hyper_modify(self):  
        # self.hyper.base_iter = 10
        # self.hyper.preTrain_epochs = 10  
        self.hyper.ppo_epochs = 1
        
        self.hyper.output_agent = False
        self.hyper.num_actions=3

        # H = 12 
        # tune H = 12  根据RL寻优得到

        self.hyper.agent_hidden_size = 128 #256
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.agent_lr = 0.001
        self.hyper.agent_step_gamma = 0.8
        self.hyper.error_rate = 0.1
        
        self.hyper.reward_alpha = 1.0  # 奖励敏感系数
        self.hyper.reward_threshold = 0.55  # 阈值 c
        self.hyper.priority_beta = 0.5  # TD误差和预测误差的权重 [0,1]
        self.hyper.lambda_min = 0.1  # 最小温度
        self.hyper.lambda_max = 2.0  # 最大温度
        self.hyper.temp_oscillation = 0.1  # 温度振荡幅度
        self.hyper.temp_frequency = 2.0  # 温度周期频率

        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  =  0.001
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.9
        self.hyper.clip_param = 0.2
        self.hyper.max_grad_norm = 0.5

        self.hyper.buffer_capacity = 100
        self.hyper.gamma = 0.9
        # self.hyper.patch_size = 6
        self.hyper.data_store_maxsize = 1000
        self.hyper.T_step = 4000

        self.hyper.dropout_rate = 0.3
        self.hyper.encoder_num_layer = 2
        
    def tuning_modify(self):
        self.tuning.error_rate = tune.quniform(0,1,0.1)
        self.tuning.agent_lr = tune.loguniform(1e-5, 1e-1)
        # self.tuning.agent_lr = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
        self.tuning.agent_step_gamma = tune.quniform(0.90,0.99,0.01)
        self.tuning.agent_hidden_size = tune.qrandint(32,512, 32)
        self.tuning.agent_dropout_rate = tune.quniform(0.1,0.9,0.1)

class RNNPSO(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/RLmodel/Multi_batch_single_variate/PSO_patch.py' #_
        self.model='PSO_patch.py'
        self.import_path = os.path.join(self.model_path, self.model)
        self.class_name = 'rnnPSO'

    
    def hyper_modify(self):  
        self.hyper.base_iter = 10
        self.hyper.preTrain_epochs = 1
        self.hyper.pso_epochs = 20
        
        self.hyper.output_agent = False
        self.hyper.num_actions=3

        # H = 12 
        # tune H = 12  根据RL寻优得到
        self.hyper.agent_hidden_size_1 = 1000
        self.hyper.agent_hidden_size = 256
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.agent_lr = 0.001
        self.hyper.agent_step_gamma = 0.8
        self.hyper.error_rate = 0.1
        # # tune H = 12  根据Type2
        # self.hyper.agent_hidden_size = 352
        # self.hyper.agent_dropout_rate = 0.7
        # self.hyper.agent_lr = 0.05
        # self.hyper.agent_step_gamma = 0.98
        # self.hyper.error_rate = 1
        # # H = 12 根据MLP寻优结果确认
        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  =  0.005
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.38

        # self.hyper.patch_size = 6

        self.hyper.dropout_rate = 0.3
        self.hyper.encoder_num_layer = 2
        self.hyper.num_particles = 20
        
    def tuning_modify(self):
        self.tuning.error_rate = tune.quniform(0,1,0.1)
        self.tuning.agent_lr = tune.loguniform(1e-5, 1e-1)
        # self.tuning.agent_lr = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
        self.tuning.agent_step_gamma = tune.quniform(0.90,0.99,0.01)
        self.tuning.agent_hidden_size = tune.qrandint(32,512, 32)
        self.tuning.agent_dropout_rate = tune.quniform(0.1,0.9,0.1)


class RNNRAN(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        # self.import_path = 'models/seq2seq/ETTlstms/RNN_patch_randn.py'
        self.model='Randn_patch.py'
        self.import_path = os.path.join(self.model_path, self.model)
        self.class_name = 'randnRNN'
    
    def hyper_modify(self):  
        self.hyper.base_iter = 10
        self.hyper.preTrain_epochs = 10  
        
        self.hyper.output_agent = False
        self.hyper.num_actions=3

        # H = 12 
        # tune H = 12  根据RL寻优得到
        self.hyper.agent_hidden_size_1 = 1000
        self.hyper.agent_hidden_size = 256
        self.hyper.agent_dropout_rate = 0.3
        self.hyper.agent_lr = 0.04227805743904207
        self.hyper.agent_step_gamma = 0.92
        self.hyper.error_rate = 0.1
        # # tune H = 12  根据Type2
        # self.hyper.agent_hidden_size = 352
        # self.hyper.agent_dropout_rate = 0.7
        # self.hyper.agent_lr = 0.05
        # self.hyper.agent_step_gamma = 0.98
        # self.hyper.error_rate = 1
        # # H = 12 根据MLP寻优结果确认
        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  =  0.005
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.38
        self.hyper.clip_param = 0.2
        self.hyper.max_grad_norm = 0.5
        self.hyper.ppo_epochs = 10
        self.hyper.buffer_capacity = 100
        self.hyper.gamma = 0.9
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
                  


class Naive11(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        # self.import_path = 'models/seq2seq/ETTlstms/RNN_patch_y01(ETTh1).py'
        self.model='Naive11_patch.py'
        self.import_path = os.path.join(self.model_path, self.model)
        self.class_name = 'RNN_Naive11' # Naive01_patch 模型下面的类
    
    def hyper_modify(self):  
        self.hyper.base_iter = 10
        self.hyper.preTrain_epochs = 10  
        
        self.hyper.output_agent = False
        self.hyper.num_actions=3

        # H = 12 
        # tune H = 12  根据RL寻优得到
        self.hyper.agent_hidden_size_1 = 1000
        self.hyper.agent_hidden_size = 256
        self.hyper.agent_dropout_rate = 0.3
        self.hyper.agent_lr = 0.01
        self.hyper.agent_step_gamma = 0.92
        self.hyper.error_rate = 0.1

        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  =  0.005
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.38
        self.hyper.clip_param = 0.2
        self.hyper.max_grad_norm = 0.5
        self.hyper.ppo_epochs = 10
        self.hyper.buffer_capacity = 100
        self.hyper.gamma = 0.9
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


class Naive01(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        # self.import_path = 'models/seq2seq/ETTlstms/RNN_patch_y01(ETTh1).py'
        self.model='Naive01_patch.py'
        self.import_path = os.path.join(self.model_path, self.model)
        self.class_name = 'RNN_Naive01' # Naive01_patch 模型下面的类
    
    def hyper_modify(self):  
        self.hyper.base_iter = 10
        self.hyper.preTrain_epochs = 10  
        
        self.hyper.output_agent = False
        self.hyper.num_actions=3

        # H = 12 
        # tune H = 12  根据RL寻优得到
        self.hyper.agent_hidden_size_1 = 1000
        self.hyper.agent_hidden_size = 256
        self.hyper.agent_dropout_rate = 0.3
        self.hyper.agent_lr = 0.01
        self.hyper.agent_step_gamma = 0.92
        self.hyper.error_rate = 0.1
        # # tune H = 12  根据Type2
        # self.hyper.agent_hidden_size = 352
        # self.hyper.agent_dropout_rate = 0.7
        # self.hyper.agent_lr = 0.05
        # self.hyper.agent_step_gamma = 0.98
        # self.hyper.error_rate = 1
        # # H = 12 根据MLP寻优结果确认
        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  =  0.005
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.38
        self.hyper.clip_param = 0.2
        self.hyper.max_grad_norm = 0.5
        self.hyper.ppo_epochs = 10
        self.hyper.buffer_capacity = 100
        self.hyper.gamma = 0.9
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


class seq2seqbaseFree(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self,):
        self.import_path = 'models/seq2seq/seq2seqbase.py'
        self.class_name = 'Seq2SeqBase'
        
    def hyper_modify(self):
        self.hyper.teacher = 'Auto-Regressive'       
    
    def tuning_modify(self):
        self.tuning.learning_rate = tune.loguniform(1e-5, 1e-1)
        self.tuning.step_gamma = tune.quniform(0.90,0.99,0.01)
        self.tuning.encoder_hidden_size = tune.qrandint(50,200, 10)
        self.tuning.encoder_num_layer = tune.qrandint(1,3,1)
        self.tuning.decoder_hidden_size = tune.qrandint( 50, 200,10)
        self.tuning.decoder_num_layer = tune.qrandint(1, 3, 1)
        self.tuning.decoder_input_size =  tune.qrandint( 50, 100,10)

class seq2seqbaseTeach_TrueV(seq2seq_base):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/seq2seqTeach.py'
        self.class_name = 'Seq2SeqTeach'
        
    def hyper_modify(self):
        self.hyper.teacher = 'TrueV' 

class seq2seqbaseTeach_MSVR(seq2seqbaseTeach_TrueV):
    def __init__(self):
        super().__init__()        
    def hyper_modify(self):
        self.hyper.teacher = 'MSVR'
         
class seq2seqbaseTeach_MLP(seq2seqbaseTeach_TrueV):
    def __init__(self):
        super().__init__()        
    def hyper_modify(self):
        self.hyper.teacher = 'MLP' 
        
class seq2seqScheduled(seq2seq_base):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/seq2seq_Scheduled.py'
        self.class_name = 'Seq2Seq_Scheduled'
        
    def hyper_modify(self):
        self.hyper.decay_approach = 'Linear'
        self.hyper.sita = 0  # [0,1)
        self.hyper.k = 1   # k=1   
        
        # tune 
        self.hyper.c = 0.43
        
    def tuning_modify(self):
        # self.hyper.epochs = 50   # fitness epoch per iter
        # self.tuner.iters = 30  # tuner search times
        # self.tuner.points_eval = None  # points_to_evaluate or not
        # self.tuner.name = 'Hyperopt_Search' #['Random_Search','Hyperopt_Search','Ax_Search']
        self.tuning.c = tune.quniform(0.01,0.99, 0.01)
        
class seq2seqProForcing(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/seq2seq/Seq2Seq_ProForcing.py'
        self.class_name = 'Seq2Seq_ProForcing'
    
    def hyper_modify(self):
        self.hyper.TrainDis_epochs = 10
        
        # tune  
        self.hyper.discriminator_hidden_size = 120
        self.hyper.discriminator_num_layer = 3
        self.hyper.discriminator_lr = 0.000564231633040006
        self.hyper.discriminator_weight_decay = 0.005

    def tuning_modify(self):
        # self.hyper.epochs = 50   # fitness epoch per iter
        # self.tuner.iters = 30  # tuner search times
        # self.tuner.points_eval = None  # points_to_evaluate or not
        # self.tuner.name = 'Hyperopt_Search' #['Random_Search','Hyperopt_Search','Ax_Search']
        self.tuning.discriminator_lr = tune.loguniform(1e-5, 1e-1)
        self.tuning.discriminator_weight_decay = tune.choice([0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05])
        self.tuning.discriminator_hidden_size = tune.qrandint(10,200, 10)
        self.tuning.discriminator_num_layer = tune.qrandint(1,5,1)
    
class seq2seqRL(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/seq2seq/seq2seq_RL.py'
        self.class_name = 'Seq2Seq_RL'
    
    def hyper_modify(self):  
        self.hyper.base_iter = 10
        self.hyper.preTrain_epochs = 1  
        
        self.hyper.output_agent = False

        self.hyper.msvr_loading = True  # loading msvr
        self.hyper.mlp_loading = True   # loading mlp
        self.hyper.seq2seq_loading = False     # loading seq2seqFree 
        self.hyper.seq2seqWTrue_loading = False   # loading se2seqTeachTrue
        
        # tune
        self.hyper.agent_hidden_size = 384
        self.hyper.agent_dropout_rate = 0.8
        self.hyper.agent_lr = 0.0076035268671218684
        self.hyper.agent_step_gamma = 0.95
        self.hyper.error_rate = 0.3
        # tune
        self.hyper.mlp_hidden_size = 400
        self.hyper.mlp_lr  = 0.0112
        self.hyper.mlp_stepLr  = 20
        self.hyper.mlp_gamma = 0.88
       
    def tuning_modify(self):
        # self.hyper.epochs = 50   # fitness epoch per iter
        # self.tuner.iters = 30  # tuner search times
        # self.tuner.points_eval = None  # points_to_evaluate or not
        # self.tuner.name = 'Hyperopt_Search' #['Random_Search','Hyperopt_Search','Ax_Search']
        self.tuning.error_rate = tune.quniform(0,1,0.1)
        self.tuning.agent_lr = tune.loguniform(1e-5, 1e-1)
        self.tuning.agent_step_gamma = tune.quniform(0.90,0.99,0.01)
        self.tuning.agent_hidden_size = tune.qrandint(32,512, 32)
        self.tuning.agent_dropout_rate = tune.quniform(0.1,0.9,0.1)
        
class msvr(nn_base):
    def base_modify(self):
        self.import_path = 'models/training/MSVR_.py'
        self.class_name = 'MSVR_'
        
        self.training = False
        self.arch = 'mlp'
        
    def hyper_init(self):
        self.hyper.kernel='rbf'
        self.hyper.degree=3 
        self.hyper.coef0=0.0
        self.hyper.tol=0.001
        # tune
        self.hyper.C= 0.15848931924611143
        self.hyper.epsilon= 1.0
        self.hyper.gamma= 0.05
        
    def tuning_init(self):
        self.tuner.iters = 30  # tuner search times
        self.tuner.points_eval = None
        self.tuner.name = 'Hyperopt_Search'  #['Random_Search','Hyperopt_Search']
        
        # total cpu cores for tuning
        self.tuner.cores = 8
        # gpu cards per trial in tune
        self.tuner.cards = 0
        # fitness repeat_num per iter
        self.tuner.epochPerIter = 1
        
        self.tuning.C = tune.choice(np.logspace(-2, 2, 41))
        self.tuning.epsilon = tune.choice(np.logspace(-4, 0, 5))
        self.tuning.gamma = tune.choice([0.05, 0.1, 0.0,0.2, 0.4])     
         
class mlp(nn_base):
    def base_modify(self):
        self.import_path = 'models/training/MLP.py'
        self.class_name = 'MLP'
        
        self.training = False
        self.arch = 'mlp'
    
    def hyper_init(self):
        # tune
        # self.hyper.hidden_size = 400
        # self.hyper.learning_rate = 0.0112
        # self.hyper.step_lr = 20
        # self.hyper.step_gamma = 0.88
        # self.hyper.epochs = 100
        
        
        # tune Valid id-18
        self.hyper.hidden_size = 450
        self.hyper.learning_rate = 0.0244
        self.hyper.step_lr = 40
        self.hyper.step_gamma = 0.46
        self.hyper.epochs = 100



    def tuning_init(self):
        self.tuner.iters = 30  # tuner search times
        self.tuner.points_eval = None  # points_to_evaluate or not
        self.tuner.name = 'Ax_Search' #['Random_Search','Hyperopt_Search','Ax_Search']
        
        # total cpu cores for tuning
        self.tuner.cores = 8
        # gpu cards per trial in tune
        self.tuner.cards = 0
        # fitness repeat_num per iter
        self.tuner.epochPerIter = 1
        
        self.tuning.hidden_size = tune.qrandint(50,600,50)
        self.tuning.learning_rate = tune.qloguniform(1e-4, 1e-1, 5e-5)
        self.tuning.step_lr = tune.qrandint(10,50, 10)
        self.tuning.step_gamma = tune.quniform(0.01,0.99, 0.01)
