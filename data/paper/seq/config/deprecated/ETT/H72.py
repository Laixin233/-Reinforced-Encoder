from ray import tune
from data.base import nn_base

from task.TaskLoader import TaskDataset, Opt

import numpy as np
import pandas as pd
import os

class ETT_data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)
        
    def info_config(self):
        self.info.normal = False
        self.info.data_path = 'data/paper/seq/real/ETT/'
        self.info.series_name = ['ETTh1','ETTh2']  
        
        # self.info.series_name = ['ETTh2']
        self.info.num_series = len(self.info.series_name) 
        self.start_time = '2018-01-01 00:00:00'
        self.info.steps = 24*7
        self.info.cov_dim = 0
        self.info.period = 24*3
        self.info.batch_size = 512
        self.info.num_variate = 7

    def sub_config(self,):
        self.seriesPack = []
        
        for i in range(self.info.num_series):
            _name = self.info.series_name[i]
            df = pd.read_csv(self.info.data_path + self.info.series_name[i] + '.csv',index_col=None,header=0)
            _start = np.where(df.values==self.start_time)[0].item()
            if self.info.num_variate == 1:
                raw_ts = df.values[_start:,-1]
            else:
                raw_ts = df.values[_start:,1:]
            sub = self.pack_dataset(raw_ts)
            sub.index = i
            sub.name = _name
            sub.H = self.info.H
            sub.merge(self.info)            

            self.seriesPack.append(sub)

class seq2seq_base(nn_base):
    def __init__(self):
        self.training = False
        self.arch = 'seq2seq'
        super().__init__()
        
    def hyper_init(self,):
        self.hyper.epochs = 100
        
        self.hyper.component = 'LSTM'
        self.hyper.optimizer = 'adam'
        
        self.hyper.encoder_hidden_size = 100
        self.hyper.encoder_num_layer = 1
        self.hyper.decoder_hidden_size = 100
        self.hyper.decoder_num_layer = 1
        self.hyper.decoder_input_size = 70
        self.hyper.learning_rate = 0.005
        self.hyper.step_gamma = 0.94
         
    def tuning_init(self,):
        # total cpu cores for tuning
        self.tuner.cores = 10
        # gpu cards per trial in tune
        self.tuner.cards = 1
        # tuner search times
        self.tuner.iters = 30
        # fitness epoch per iter
        self.tuner.epochPerIter = 1
              
class seq2seqbaseFree(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self,):
        self.import_path = 'models/seq2seq/seq2seqbase.py'
        self.class_name = 'Seq2SeqBase'
        
    def hyper_modify(self):
        self.hyper.teacher = 'Auto-Regressive'       
    
    def tuning_modify(self):
        self.tuning.learning_rate = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
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
        
        self.hyper.c = 0.75
        
    def tuning_modify(self):
        # tuner search times
        self.tuner.iters = 30
        # fitness epoch per iter
        # self.hyper.epochs = 100
        self.tuning.c = tune.quniform(0.01,0.99, 0.01)
        
class seq2seqProForcing(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/seq2seq/Seq2Seq_ProForcing.py'
        self.class_name = 'Seq2Seq_ProForcing'
    
    def hyper_modify(self):
        self.hyper.TrainDis_epochs = 10
                
        self.hyper.discriminator_hidden_size = 140
        self.hyper.discriminator_num_layer = 2
        self.hyper.discriminator_lr = 0.0005
        self.hyper.discriminator_weight_decay = 0.001

    def tuning_modify(self):
        # tuner search times
        self.tuner.iters = 30
        # fitness epoch per iter
        self.hyper.epochs = 50
        self.tuning.discriminator_lr = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
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
        self.hyper.preTrain_epochs = 10  
        
        self.hyper.agent_T_agent = True
        self.hyper.output_agent = False
        
        self.hyper.msvr_loading = True  # loading msvr
        self.hyper.mlp_loading = True   # loading mlp
        self.hyper.seq2seq_loading = True     # loading seq2seqFree 
        self.hyper.seq2seqWTrue_loading = False   # loading se2seqTeachTrue
        
        self.hyper.agent_hidden_size = 96
        self.hyper.agent_dropout_rate = 0.2
        self.hyper.agent_lr = 0.005
        
        self.hyper.agent_step_gamma = 0.95
        self.hyper.mlp_hidden_size = 150
        self.hyper.mlp_lr  = 0.005
        self.hyper.mlp_stepLr  = 20
        self.hyper.mlp_gamma = 0.51
       
    def tuning_modify(self):
        # tuner search times
        self.tuner.iters = 30
        # fitness epoch per iter
        self.hyper.epochs = 50
        
        self.tuning.agent_lr = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
        self.tuning.agent_step_gamma = tune.quniform(0.90,0.99,0.01)
        self.tuning.agent_hidden_size = tune.qrandint(32,512, 32)
        self.tuning.agent_dropout_rate = tune.quniform(0.1,0.9,0.1)
        
class msvr(nn_base):
    def base_modify(self):
        self.import_path = 'models/training/MSVR_.py'
        self.class_name = 'MSVR_'
        
        self.training = False
        self.arch = 'mlp'
    
    def tuning_init(self):
        # total cpu cores for tuning
        self.tuner.cores = 16
        # gpu cards per trial in tune
        self.tuner.cards = 0
        # tuner search times
        self.tuner.iters = 30
        # fitness epoch per iter
        self.tuner.epochPerIter = 1
        
        self.tuning.C = tune.choice(np.logspace(-2, 2, 32))
        self.tuning.epsilon = tune.choice(np.logspace(-4, 0, 4))
        self.tuning.gamma = tune.choice([0.05, 0.1, 0.2, 0.4])
        
    def hyper_init(self):
        self.hyper.kernel='rbf'
        self.hyper.degree=3 
        self.hyper.coef0=0.0
        self.hyper.tol=0.001
        self.hyper.C= 1.0
        self.hyper.epsilon=0.1
        self.hyper.gamma= None
  
class mlp(nn_base):
    def base_modify(self):
        self.import_path = 'models/training/MLP.py'
        self.class_name = 'MLP'
        
        self.training = False
        self.arch = 'mlp'
    
    def hyper_init(self):
        self.hyper.hidden_size = 100
        self.hyper.learning_rate = 0.01
        self.hyper.step_lr = 40
        self.hyper.step_gamma = 0.47
        
        self.hyper.epochs = 100
        
    def tuning_init(self):
        # tuner search times
        self.tuner.iters = 30
        # fitness epoch per iter
        self.hyper.epochs = 100
        
        # total cpu cores for tuning
        self.tuner.cores = 10
        # gpu cards per trial in tune
        self.tuner.cards = 1
        # fitness repeat_num per iter
        self.tuner.epochPerIter = 1
        self.tuning.hidden_size = tune.qrandint(50,600,50)
        self.tuning.learning_rate = tune.qloguniform(1e-4, 1e-1, 5e-5)
        self.tuning.step_lr = tune.qrandint(10,50, 10)
        self.tuning.step_gamma = tune.quniform(0.01,0.99, 0.01)