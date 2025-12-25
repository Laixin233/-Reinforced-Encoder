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
        self.info.data_path = 'data/paper/seq/real/ili/ILI.csv'
        # self.info.series_name = ['south','north']
        # self.info.num_series = 2
        self.info.series_name = ['south']
        self.info.num_series = 1
        self.info.cov_dim = 0
        self.info.steps = 26
        self.info.input_dim = 1
        self.info.period = 4
        self.info.num_variate = 1

    def sub_config(self,):
        self.seriesPack = []
            
        for i, name in enumerate(self.info.series_name):
            df = pd.read_csv(self.info.data_path, header=0)
            data = df[name+'_ILI']
            if data.isnull().any():
                data= data.interpolate()
            raw_ts = data.values.reshape(-1, )
            # np.save(npy_file, raw_ts)
                    
            sub = self.pack_dataset(raw_ts)
            sub.index = i
            sub.name = name
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
        
        # H = 4
        self.hyper.encoder_hidden_size = 150
        self.hyper.encoder_num_layer = 1
        self.hyper.decoder_hidden_size = 150
        self.hyper.decoder_num_layer = 1
        self.hyper.decoder_input_size = 80
        self.hyper.learning_rate = 0.005
        self.hyper.step_gamma = 0.99
    
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
        self.hyper.teach_Force = False
        self.hyper.test_teach_Force  = False   
    
    def tuning_modify(self):
        self.tuning.learning_rate = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
        # self.tuning.step_gamma = tune.choice([0.90,0.95,0.97,0.99])
        # H 12
        self.tuning.step_gamma = tune.quniform(0.90,0.99,0.01)
        
        self.tuning.encoder_hidden_size = tune.qrandint(50,200, 10)
        self.tuning.encoder_num_layer = tune.qrandint(1,3,1)
        self.tuning.decoder_hidden_size = tune.qrandint( 50, 200,10)
        self.tuning.decoder_num_layer = tune.qrandint(1, 3, 1)
        self.tuning.decoder_input_size =  tune.qrandint( 50, 100,10)

class seq2seqbaseTeachF(seq2seq_base):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/seq2seqbase.py'
        self.class_name = 'Seq2SeqBase'
        
    def hyper_modify(self):
        self.hyper.teach_Force = True    
        self.hyper.test_teach_Force  = False
        
class seq2seqRL(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/seq2seq/seq2seq_RL.py'
        self.class_name = 'Seq2Seq_RL'
    
    def hyper_modify(self):
        self.hyper.base_iter = 10
        self.hyper.preTrain_epochs = 10   # v1 = 10
        
        self.hyper.agent_T_agent = True
        self.hyper.output_agent = False

        # H=4
        self.hyper.agent_hidden_size = 288
        self.hyper.agent_dropout_rate = 0.3
        self.hyper.agent_lr = 0.0005
        self.hyper.agent_step_gamma = 0.93    
    
    def tuning_modify(self):
        # fitness epoch per iter
        # self.hyper.epochs = 100
        # tuner search times
        self.tuner.iters = 15

        self.tuning.agent_lr = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
        self.tuning.agent_step_gamma = tune.quniform(0.90,0.99,0.01)
        self.tuning.agent_hidden_size = tune.qrandint(32,512, 32)
        self.tuning.agent_dropout_rate = tune.quniform(0.1,0.9,0.1)