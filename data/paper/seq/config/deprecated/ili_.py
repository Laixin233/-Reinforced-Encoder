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
        self.info.data_path = 'data/paper/seq/ili/ILI.csv'
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
            
        hostDir_path = 'data/paper/esn/ili/'
        for i, name in enumerate(self.info.series_name):
            # npy_file = os.path.join(hostDir_path, '{}.npy'.format(name))
            # if os.path.exists(npy_file):
            #     raw_ts = np.load(npy_file)
            # else:
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
        
        # H = 8
        # self.hyper.encoder_hidden_size = 150
        # self.hyper.encoder_num_layer = 1
        # self.hyper.decoder_hidden_size = 150
        # self.hyper.decoder_num_layer = 1
        # self.hyper.decoder_input_size = 80
        # self.hyper.learning_rate = 0.005
        # self.hyper.step_gamma = 0.99
        
        # H = 10
        # self.hyper.encoder_hidden_size = 70
        # self.hyper.encoder_num_layer = 1
        # self.hyper.decoder_hidden_size = 180
        # self.hyper.decoder_num_layer = 2
        # self.hyper.decoder_input_size = 60
        # self.hyper.learning_rate = 0.05
        # self.hyper.step_gamma = 0.95
        
        # tune H = 12
        # self.hyper.encoder_hidden_size = 140
        # self.hyper.encoder_num_layer = 2
        # self.hyper.decoder_hidden_size = 80
        # self.hyper.decoder_num_layer = 1
        # self.hyper.decoder_input_size = 90
        # self.hyper.learning_rate = 0.05
        # self.hyper.step_gamma = 0.91
    
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
         # tune
        
        # # H = 4
        # self.hyper.c = 0.82
        
        # H = 8
        # self.hyper.c = 0.9
        
        # H = 10
        # self.hyper.c = 0.83
        
        # tune H = 12
        self.hyper.c = 0.29
        
        
        # self.hyper.c = 1/(self.hyper.epochs - 20)
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
        self.hyper.TrainDis_epochs = 10
        
        # H = 4 tune   
        # self.hyper.discriminator_hidden_size = 40
        # self.hyper.discriminator_num_layer = 1
        # self.hyper.discriminator_lr = 0.05
        # self.hyper.discriminator_weight_decay =  0.001
        
        # H = 8 tune   
        # self.hyper.discriminator_hidden_size = 80
        # self.hyper.discriminator_num_layer = 2
        # self.hyper.discriminator_lr = 0.001
        # self.hyper.discriminator_weight_decay =  0.001
        
        # H = 10 tune   
        # self.hyper.discriminator_hidden_size = 110
        # self.hyper.discriminator_num_layer = 4
        # self.hyper.discriminator_lr = 0.0005
        # self.hyper.discriminator_weight_decay =  0.0001
        
        # H = 12 tune   
        self.hyper.discriminator_hidden_size = 160
        self.hyper.discriminator_num_layer = 3
        self.hyper.discriminator_lr = 0.0005
        self.hyper.discriminator_weight_decay =  0.0005

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
        self.hyper.epochs = 20
        
        # H =4 tune
        # # H = 4 根据Type2寻优得到
        # self.hyper.agent_hidden_size = 288
        # self.hyper.agent_dropout_rate = 0.3
        # self.hyper.agent_lr = 0.0005
        # self.hyper.agent_step_gamma = 0.93
        # # H = 4 根据MLP寻优结果确认
        # self.hyper.mlp_hidden_size = 250
        # self.hyper.mlp_lr  = 0.01
        # self.hyper.mlp_stepLr  = 30
        # self.hyper.mlp_gamma = 0.67
        
        # H = 8 根据Type2寻优得到
        # self.hyper.agent_hidden_size = 96
        # self.hyper.agent_dropout_rate = 0.3
        # self.hyper.agent_lr = 0.001
        # self.hyper.agent_step_gamma = 0.98
        # # H = 8 second tune根据Type2寻优得到
        # self.hyper.agent_hidden_size = 96
        # self.hyper.agent_dropout_rate = 0.2
        # self.hyper.agent_lr = 0.005
        # self.hyper.agent_step_gamma = 0.95
        # # H = 8 根据MLP寻优结果确认 tune
        # self.hyper.mlp_hidden_size = 450
        # self.hyper.mlp_lr  = 0.005
        # self.hyper.mlp_stepLr  = 30
        # self.hyper.mlp_gamma = 0.6
        
        # # H = 10 tune
        # # H = 10 根据Type2寻优得到
        # self.hyper.agent_hidden_size = 320
        # self.hyper.agent_dropout_rate = 0.6
        # self.hyper.agent_lr = 0.05
        # self.hyper.agent_step_gamma = 0.93
        # # H = 10 根据MLP寻优结果确认
        # self.hyper.mlp_hidden_size = 250
        # self.hyper.mlp_lr  =  0.005
        # self.hyper.mlp_stepLr  = 30
        # self.hyper.mlp_gamma = 0.38
        
        # H = 12 
        # tune H = 12
        self.hyper.agent_hidden_size = 352
        self.hyper.agent_dropout_rate = 0.7
        self.hyper.agent_lr = 0.05
        self.hyper.agent_step_gamma = 0.98
        # H = 12 根据MLP寻优结果确认
        self.hyper.mlp_hidden_size = 250
        self.hyper.mlp_lr  =  0.005
        self.hyper.mlp_stepLr  = 30
        self.hyper.mlp_gamma = 0.38
        
    
    
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

class seq2seqRLType3(seq2seqRL):
    def __init__(self):
        super().__init__()
        self.hyper.agent_T_agent = True
        self.hyper.seq2seq_T_agent = True
        self.hyper.output_agent = True

class seq2seqRLPreT(seq2seqRLType2):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/seq2seq/seq2seq_RL_preTrain.py'
        self.class_name = 'Seq2Seq_RL'
        
class Seq2Seq_new_RL(seq2seqRL):
    def __init__(self):
        super().__init__()
        self.hyper.agent_T_agent = True
        self.hyper.seq2seq_T_agent = False
        self.hyper.output_agent = False
        
    def base_modify(self):
        self.import_path = 'models/seq2seq/seq2seq_new_RL.py'
        self.class_name = 'Seq2Seq_RL'
        
class transformer(nn_base):
    def __init__(self):
        self.training = False
        self.arch = 'seq2seq'
        super().__init__()
        
    def hyper_init(self,):
        self.hyper.epochs = 100
        
        self.hyper.optimizer = 'adam'
        self.hyper.learning_rate = 0.01
        self.hyper.weight_decay = 0.0001
        
        self.hyper.encoder_num_layer = 6
        self.hyper.decoder_num_layer = 6
        
        self.hyper.encoder_input_size = 1
        self.hyper.decoder_input_size = 1
        
        self.hyper.channels = 256     # channels需是nhead的整数倍
        self.hyper.dropout = 0.2
        self.hyper.nhead = 8

        
class transformerbaseFree_default(transformer):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/transformerbase.py'
        self.class_name = 'TransformerBase'
        
    def hyper_modify(self):
        self.hyper.teach_Force = False
        
class transformerbaseTeachF_default(transformer):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/transformerbase.py'
        self.class_name = 'TransformerBase'
        
    def hyper_modify(self):
        self.hyper.teach_Force = True

class transformerRL(transformer):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/transformer_RL.py'
        self.class_name = 'Transformer_RL'
        
    def hyper_modify(self):
        self.hyper.preTrain_epochs = 10
        
        self.hyper.agent_hidden_size = 128
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.mlp_hidden_size = 400
        self.hyper.mlp_lr  = 0.0001
        self.hyper.mlp_stepLr  = 20
        self.hyper.mlp_gamma = 0.9
        
        self.hyper.environment_lr = 0.005
        self.hyper.environment_gamma = 0.99
        
        self.hyper.agent_lr = 0.05
        self.hyper.agent_step_gamma = 0.99

class transformerRLType2(transformerRL):
    def __init__(self):
        super().__init__()

        self.hyper.agent_T_agent = True
        self.hyper.seq2seq_T_agent = False
        self.hyper.output_agent = False
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
        
        # H = 4，12
        self.hyper.C= 1.0
        self.hyper.epsilon=0.1
        self.hyper.gamma= None
        
    def tuning_init(self):
        # total cpu cores for tuning
        self.tuner.cores = 10
        # gpu cards per trial in tune
        self.tuner.cards = 1
        # tuner search times
        self.tuner.iters = 30
        # fitness epoch per iter
        self.tuner.epochPerIter = 1
        
        
        self.tuning.C = tune.choice(np.logspace(-2, 2, 32))
        self.tuning.epsilon = tune.choice(np.logspace(-4, 0, 4))
        self.tuning.gamma = tune.choice([0.05, 0.1, 0.2, 0.4])
        
class mlp(nn_base):
    def base_modify(self):
        self.import_path = 'models/training/MLP.py'
        self.class_name = 'MLP'
        
        self.training = False
        self.arch = 'mlp'
    
    def hyper_init(self):
        # tune
        # H = 4
        # self.hyper.hidden_size = 250
        # self.hyper.learning_rate = 0.01
        # self.hyper.step_lr = 30
        # self.hyper.step_gamma = 0.67
        
        # H = 8  tune second
        # self.hyper.hidden_size = 450
        # self.hyper.learning_rate =  0.005
        # self.hyper.step_lr = 30
        # self.hyper.step_gamma = 0.6
        
        # H = 10
        # self.hyper.hidden_size = 250
        # self.hyper.learning_rate = 0.005
        # self.hyper.step_lr = 30
        # self.hyper.step_gamma = 0.38
        
        # tune H = 12
        self.hyper.hidden_size = 250
        self.hyper.learning_rate = 0.005
        self.hyper.step_lr = 30
        self.hyper.step_gamma = 0.38
        
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
        # self.tuning.learning_rate = tune.choice([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1])
        self.tuning.learning_rate = tune.choice([0.0001,0.0005,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
        self.tuning.step_lr = tune.qrandint(10,50, 10)
        self.tuning.step_gamma = tune.quniform(0.01,0.99, 0.01)