# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
# from numpy.lib.function_base import select
from ray import tune
from data.base import nn_base

from task.TaskLoader import TaskDataset, Opt
import numpy as np

class mg_data(TaskDataset):
    def __init__(self, args):
        super().__init__(args)
        
    def info_config(self):
        self.info.normal = False
        self.info.num_series = 1
        self.info.steps = 200
        self.info.input_dim = 1
        self.info.cov_dim = 0
        self.info.period = 17
        self.info.batch_size = 512  # transformer为64，其他为512
        self.info.num_variate = 1
    
    def sub_config(self,):
        self.seriesPack = []
        
        for i in range(self.info.num_series):
            raw_ts = np.load(
                'data/paper/seq/mg/mg.npy').reshape(-1,)
            sub = self.pack_dataset(raw_ts)
            sub.index = i
            sub.name = 'mg17'
            sub.H = self.info.H
            sub.merge(self.info)
            # Attention! sub.H can be different among multiple series in some cases, e.g. NN3 dataset.

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
        
        # H = 17
        # self.hyper.encoder_hidden_size = 150
        # self.hyper.encoder_num_layer = 1
        # self.hyper.decoder_hidden_size = 150
        # self.hyper.decoder_num_layer = 1
        # self.hyper.decoder_input_size = 80
        # self.hyper.learning_rate = 0.005
        # self.hyper.step_gamma = 0.99
        
        
        # H = 84
        self.hyper.encoder_hidden_size = 150
        self.hyper.encoder_num_layer = 1
        self.hyper.decoder_hidden_size = 130
        self.hyper.decoder_num_layer = 2
        self.hyper.decoder_input_size = 70
        self.hyper.learning_rate = 0.01
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
        self.tuning.step_gamma = tune.choice([0.90,0.95,0.97,0.99])
        
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
        
class seq2seqScheduled(seq2seq_base):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/seq2seq_Scheduled.py'
        self.class_name = 'Seq2Seq_Scheduled'
        
    def hyper_modify(self):
        # self.hyper.decay_approach = 'Exponential'
        # self.hyper.k = 0.8  # k<1
        # self.hyper.sita = 0
        # self.hyper.c = 0
        
        # self.hyper.decay_approach = 'Exponential'
        # self.hyper.k = 5   # k>=1
        # self.hyper.sita = 0
        # self.hyper.c = 0
        
        self.hyper.decay_approach = 'Linear'
         # tune   H = 17
        self.hyper.c = 0.15
        
        
        self.hyper.sita = 0  # [0,1)
        self.hyper.k = 1   # k=1    
        
    def tuning_modify(self):
        self.tuning.c = tune.quniform(0.01,0.99, 0.01)
        

class seq2seqProForcing(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/seq2seq/Seq2Seq_ProForcing.py'
        self.class_name = 'Seq2Seq_ProForcing'
    
    def hyper_modify(self):
        # tune  H = 17
        self.hyper.discriminator_hidden_size = 20
        self.hyper.discriminator_num_layer = 4
        self.hyper.discriminator_lr = 0.05
        self.hyper.discriminator_weight_decay = 0.01
        
        self.hyper.TrainDis_epochs = 10

    def tuning_modify(self):
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
        self.hyper.preTrain_epochs = 10
        self.hyper.base_iter = 10
        
        # tune H = 17 tril_id 00009
        self.hyper.agent_hidden_size = 320
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.agent_lr = 0.05
        self.hyper.agent_step_gamma = 0.93
        
        # H = 17 根据MLP寻优结果确认
        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  = 0.01
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.67
        
        # H = 84 根据MLP寻优结果确认
        # self.hyper.mlp_hidden_size = 250
        # self.hyper.mlp_lr  = 0.01
        # self.hyper.mlp_stepLr  = 30
        # self.hyper.mlp_gamma = 0.67
    
    def tuning_modify(self):
        self.tuning.agent_lr = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
        self.tuning.agent_step_gamma = tune.quniform(0.90,0.99,0.01)
        self.tuning.agent_hidden_size = tune.qrandint(32,512, 32)
        self.tuning.agent_dropout_rate = tune.quniform(0.1,0.9,0.1)

class seq2seqRLType1(seq2seqRL):
    def __init__(self):
        super().__init__()
        self.hyper.agent_T_agent = True
        self.hyper.seq2seq_T_agent = False
        self.hyper.output_agent = True
        
             
class seq2seqRLType2(seq2seqRL):
    def __init__(self):
        super().__init__()
        self.hyper.agent_T_agent = True
        self.hyper.seq2seq_T_agent = False
        self.hyper.output_agent = False
        
class mlp(nn_base):
    def base_modify(self):
        self.import_path = 'models/training/MLP.py'
        self.class_name = 'MLP'
        
        self.training = False
        self.arch = 'mlp'
    
    def hyper_init(self):
        # tune H=17
        # self.hyper.hidden_size = 250
        # self.hyper.learning_rate = 0.01
        # self.hyper.step_lr = 30
        # self.hyper.step_gamma = 0.67
        
        # tune H=84
        self.hyper.hidden_size = 250
        self.hyper.learning_rate = 0.005
        self.hyper.step_lr = 40
        self.hyper.step_gamma = 0.39
        
        self.hyper.epochs = 100
        
    def tuning_init(self):
        # total cpu cores for tuning
        self.tuner.cores = 10
        # gpu cards per trial in tune
        self.tuner.cards = 1
        # tuner search times
        self.tuner.iters = 30
        # fitness epoch per iter
        self.tuner.epochPerIter = 1
        
        
        self.tuning.hidden_size = tune.qrandint(50,600, 50)
        self.tuning.learning_rate = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
        self.tuning.step_lr = tune.qrandint(10,50, 10)
        self.tuning.step_gamma = tune.quniform(0.01,0.99, 0.01)