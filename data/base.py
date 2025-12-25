# from ray import tune
from matplotlib.pyplot import cla
from task.TaskLoader import Opt
from ray import tune

class nn_base(Opt):
    def __init__(self):
        super().__init__()
        self.hyper = Opt()
        
        self.tuner = Opt()
        self.tuning = Opt()

        self.hyper_init()
        self.tuning_init()

        self.base_modify()
        self.hyper_modify()

        self.tuning_modify()
        self.ablation_modify()
        
        
        self.common_process()

    def hyper_init(self,):
        pass

    def tuning_init(self,):
        # total cpu cores for tuning
        self.tuner.cores = 10
        # gpu cards per trial in tune
        self.tuner.cards = 1
        # tuner search times
        self.tuner.iters = 16
        # fitness epoch per iter
        self.tuner.epochPerIter = 1

    def base_modify(self,):
        pass

    def hyper_modify(self,):
        pass

    def tuning_modify(self):
        pass

    def ablation_modify(self):
        pass

    def common_process(self,):
        if "import_path" in self.dict:
            self.import_path = self.import_path.replace(
            '.py', '').replace('/', '.')




class transformer_base(nn_base):
    def __init__(self):
        self.training = False
        self.arch = 'seq2seq'
        super().__init__()
        
    def hyper_init(self,):
        self.hyper.epochs = 100
        
        self.hyper.optimizer = 'adam'
        self.hyper.learning_rate = 0.01
        self.hyper.weight_decay = 0.00001
        
        self.hyper.encoder_num_layer = 2
        self.hyper.decoder_num_layer = 2
        
        self.hyper.encoder_input_size = 1
        self.hyper.decoder_input_size = 1
        
        self.hyper.channels = 24     # channels需是nhead的整数倍
        self.hyper.dropout = 0.2
        self.hyper.nhead = 4
    
    def tuning_init(self):
        self.tuning.optimizer =  tune.choice(['adam','adagrad','SGD'])
        self.tuning.learning_rate = tune.uniform(0.005, 0.015)
        self.tuning.weight_scaling = tune.choice([0.00001,0.0001,0.001])
        
        self.tuning.encoder_num_layer = tune.qrandint(1, 10, 1)
        self.tuning.decoder_num_layer = tune.qrandint(1, 10, 1)
        self.tuning.channels = tune.qrandint(1, 10, 1)
        self.tuning.dropout  = tune.uniform(0.1, 0.2)
        self.tuning.nhead = tune.qrandint(2, 8, 1)
        
class transformerbaseFree_default(transformer_base):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/transformerbase.py'
        self.class_name = 'TransformerBase'
        
    def hyper_modify(self):
        self.hyper.teach_Force = False
        
class transformerbaseTeachF_default(transformer_base):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/transformerbase.py'
        self.class_name = 'TransformerBase'
        
    def hyper_modify(self):
        self.hyper.teach_Force = True

class transformerRL_default(transformer_base):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/transformer_RL.py'
        self.class_name = 'Transformer_RL'
        
    def hyper_modify(self):
        self.hyper.preTrain_epochs = 1

class seq2seq_base(nn_base):
    def __init__(self):
        self.training = False
        self.arch = 'seq2seq'
        super().__init__()
        
    def hyper_init(self,):
        self.hyper.epochs = 20
        
        self.hyper.component = 'RNN'
        
        self.hyper.optimizer = 'adam'
        self.hyper.learning_rate = 0.0005
        self.hyper.step_gamma = 0.95
        
        self.hyper.encoder_hidden_size = 100
        self.hyper.encoder_num_layer = 1
        self.hyper.decoder_hidden_size = 100
        self.hyper.decoder_num_layer = 1
        self.hyper.decoder_input_size = 100
    
    def tuning_init(self):
        # total cpu cores for tuning
        self.tuner.cores = 10
        # gpu cards per trial in tune
        self.tuner.cards = 1
        # tuner search times
        self.tuner.iters = 16
        # fitness epoch per iter
        self.tuner.epochPerIter = 10
        
        # self.tuning.component =  tune.choice(['LSTM','RNN','GRU'])
        self.tuning.optimizer =  tune.choice(["adam","adagrad","SGD"])
        self.tuning.learning_rate = tune.uniform(0.005, 0.015)
        self.tuning.weight_decay = tune.choice([0.00001,0.0001,0.001])        
        
        self.tuning.encoder_hidden_size = tune.qrandint(50, 200, 10)
        self.tuning.encoder_num_layer = tune.qrandint(1, 5, 1)
        self.tuning.decoder_hidden_size = tune.qrandint(50, 200, 10)
        self.tuning.decoder_num_layer = tune.qrandint(1, 5, 1)
        self.tuning.decoder_input_size = tune.qrandint(50, 200, 10)

class RNNRL_default(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/seq2seq/RNN_RL.py'
        self.class_name = 'RNN_RL'
    
    def hyper_modify(self):
        self.hyper.preTrain_epochs = 10
        self.hyper.agent_hidden_size = 128
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.mlp_hidden_size = 100
        
        self.hyper.agent_T_agent = True
        self.hyper.seq2seq_T_agent = False
        self.hyper.output_agent = False
    
    def tuning_init(self):
        # total cpu cores for tuning
        self.tuner.cores = 10
        # gpu cards per trial in tune
        self.tuner.cards = 1
        # tuner search times
        self.tuner.iters = 16
        # fitness epoch per iter
        self.tuner.epochPerIter = 10
        
        self.tuning.optimizer =  tune.choice(["adam","adagrad","SGD"])
        self.tuning.learning_rate = tune.uniform(0.005, 0.015)
        self.tuning.weight_decay = tune.choice([0.00001,0.0001,0.001])  
        
        self.tuning.preTrain_epochs = tune.qrandint(5, 100, 5)
        self.tuning.agent_hidden_size = tune.qrandint(50, 200, 10)
        self.tuning.agent_dropout_rate = tune.uniform(0.1, 0.9)
        self.tuning.mlp_hidden_size = tune.qrandint(10, 100, 10)



class seq2seqbase_default(seq2seq_base):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/seq2seqbase.py'
        self.class_name = 'Seq2SeqBase'
        
    def hyper_modify(self):
        self.hyper.teach_Force = None
              
class seq2seqbaseFree_default(seq2seq_base):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/seq2seqbase.py'
        self.class_name = 'Seq2SeqBase'
        
    def hyper_modify(self):
        self.hyper.teach_Force = False

class seq2seqbaseTeachF_default(seq2seq_base):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/seq2seqbase.py'
        self.class_name = 'Seq2SeqBase'
        
    def hyper_modify(self):
        self.hyper.teach_Force = True       
        
class seq2seqScheduled_default(seq2seq_base):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/seq2seq/seq2seq_Scheduled.py'
        self.class_name = 'Seq2Seq_Scheduled'
        
    def hyper_modify(self):
        self.hyper.decay_approach = 'Linear'
        self.hyper.c = 1/(self.hyper.epochs - 20)
        self.hyper.sita = 0  # [0,1)
        self.hyper.k = 1   # k=1    
        
        # self.hyper.decay_approach = 'Exponential'
        # self.hyper.k = 0.8  # k<1
        # self.hyper.sita = 0
        # self.hyper.c = 0
        
        # self.hyper.decay_approach = 'Exponential'
        # self.hyper.k = 5   # k>=1
        # self.hyper.sita = 0
        # self.hyper.c = 0
        

class seq2seqProForcing_default(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/seq2seq/Seq2Seq_ProForcing.py'
        self.class_name = 'Seq2Seq_ProForcing'
    
    def hyper_modify(self):
        self.hyper.discriminator_hidden_size = 50
        self.hyper.discriminator_num_layer = 2
        self.hyper.TrainDis_epochs = 5
    
    def tuning_init(self):
        # total cpu cores for tuning
        self.tuner.cores = 10
        # gpu cards per trial in tune
        self.tuner.cards = 1
        # tuner search times
        self.tuner.iters = 16
        # fitness epoch per iter
        self.tuner.epochPerIter = 10
        self.tuning.optimizer =  tune.choice(["adam","adagrad","SGD"])
        self.tuning.learning_rate = tune.uniform(0.005, 0.015)
        self.tuning.weight_decay = tune.choice([0.00001,0.0001,0.001])  
        
        self.tuning.discriminator_hidden_size = tune.qrandint(10, 200, 10)
        self.tuning.discriminator_num_layer = tune.qrandint(1, 10, 1)
        self.tuning.TrainDis_epochs=  tune.qrandint(10, 100, 5)

class seq2seqRL_default(seq2seq_base):
    def __init__(self):
        super().__init__()
        
    def base_modify(self):
        self.import_path = 'models/seq2seq/seq2seq_RL.py'
        self.class_name = 'Seq2Seq_RL'
    
    def hyper_modify(self):
        self.hyper.preTrain_epochs = 10
        self.hyper.agent_hidden_size = 128
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.mlp_hidden_size = 100
        
        self.hyper.agent_T_agent = True
        self.hyper.seq2seq_T_agent = False
        self.hyper.output_agent = False
    
    def tuning_init(self):
        # total cpu cores for tuning
        self.tuner.cores = 10
        # gpu cards per trial in tune
        self.tuner.cards = 1
        # tuner search times
        self.tuner.iters = 16
        # fitness epoch per iter
        self.tuner.epochPerIter = 10
        
        self.tuning.optimizer =  tune.choice(["adam","adagrad","SGD"])
        self.tuning.learning_rate = tune.uniform(0.005, 0.015)
        self.tuning.weight_decay = tune.choice([0.00001,0.0001,0.001])  
        
        self.tuning.preTrain_epochs = tune.qrandint(5, 100, 5)
        self.tuning.agent_hidden_size = tune.qrandint(50, 200, 10)
        self.tuning.agent_dropout_rate = tune.uniform(0.1, 0.9)
        self.tuning.mlp_hidden_size = tune.qrandint(10, 100, 10)

class seq2seqRLType1_default(seq2seqRL_default):
    def hyper_modify(self):
        self.hyper.preTrain_epochs = 10
        self.hyper.agent_hidden_size = 128
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.mlp_hidden_size = 100
        
        self.hyper.agent_T_agent = True
        self.hyper.seq2seq_T_agent = False
        self.hyper.output_agent = True
class seq2seqRLType2_default(seq2seqRL_default):
    def hyper_modify(self):
        self.hyper.preTrain_epochs = 10
        self.hyper.agent_hidden_size = 128
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.mlp_hidden_size = 100
        
        self.hyper.agent_T_agent = True
        self.hyper.seq2seq_T_agent = False
        self.hyper.output_agent = False
class seq2seqRLType3_default(seq2seqRL_default):
    def hyper_modify(self):
        self.hyper.preTrain_epochs = 10
        self.hyper.agent_hidden_size = 128
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.mlp_hidden_size = 100
        
        self.hyper.agent_T_agent = True
        self.hyper.seq2seq_T_agent = True
        self.hyper.output_agent = True
class seq2seqRLType4_default(seq2seqRL_default):
    def hyper_modify(self):
        self.hyper.preTrain_epochs = 10
        self.hyper.agent_hidden_size = 128
        self.hyper.agent_dropout_rate = 0.6
        self.hyper.mlp_hidden_size = 100
        
        self.hyper.agent_T_agent = False
        self.hyper.seq2seq_T_agent = False
        self.hyper.output_agent = False













class mlp_default(nn_base):
    def base_modify(self):
        self.import_path = 'models/training/MLP.py'
        self.class_name = 'MLP'
        
        self.training = False
        self.arch = 'mlp'
    
    def hyper_init(self):
        self.hyper.epochs = 100
        
        self.hyper.hidden_size = 400
        self.hyper.learning_rate = 0.01
        self.hyper.step_lr = 20
    
    def tuning_modify(self):   
        self.tuning.hidden_size = tune.qrandint(50, 500, 10)
        self.tuning.learning_rate = tune.uniform(0.005, 0.015)
        self.tuning.step_lr = tune.qrandint(10, 20, 5)
        
class arima_default(nn_base):
    def base_modify(self):
        self.import_path = 'models/statistical/arima.py'
        self.class_name = 'ARIMA'
        
        self.training = False
        self.arch = 'mlp'
    
    def hyper_init(self):
        self.hyper.refit = False
        
class msvr_default(nn_base):
    def base_modify(self):
        self.import_path = 'models/training/MSVR_.py'
        self.class_name = 'MSVR_'
        
        self.training = False
        self.arch = 'mlp'
    
    def hyper_init(self):
        self.hyper.kernel='rbf'
        self.hyper.degree=3 
        self.hyper.gamma=None
        self.hyper.coef0=0.0
        self.hyper.tol=0.001
        self.hyper.C=1.0
        self.hyper.epsilon=0.1
        
class naivelast_default(nn_base):
    def base_modify(self):
        self.import_path = 'models/statistical/naive.py'
        self.class_name = 'Naive'
        
        self.training = False
        self.arch = 'mlp'
    
    def hyper_init(self):
        self.hyper._method = 'last'   # 'last';'avg';'period'

class naiveavg_default(nn_base):
    def base_modify(self):
        self.import_path = 'models/statistical/naive.py'
        self.class_name = 'Naive'
        
        self.training = False
        self.arch = 'mlp'
    
    def hyper_init(self):
        self.hyper._method = 'avg'   # 'last';'avg';'period'
        
















        
        
        

class esn_base(nn_base):
    def __init__(self):
        self.training = False
        self.arch = 'rnn'
        super().__init__()

    def hyper_init(self,):        
        self.hyper.leaky_r = 0.5
        self.hyper.discard_steps = 0
        self.hyper.hidden_size = 400
        self.hyper.lambda_reg = 0
        self.hyper.nonlinearity = 'tanh'
        self.hyper.read_hidden = 'last'
        self.hyper.iw_bound = (-0.1, 0.1)
        self.hyper.hw_bound = (-1, 1)
        self.hyper.weight_scaling = 0.9
        self.hyper.init = 'vanilla'
        self.hyper.fc_io = 'on'

class esn_default(esn_base):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/ESN.py'
        self.class_name = 'EchoStateNetwork'
        
    def hyper_modify(self):
        self.hyper.hidden_size = 800

    def tuning_modify(self):
        self.tuning.leaky_r = tune.uniform(0.49, 1)
        self.tuning.weight_scaling = tune.uniform(0.6, 0.99)
        self.tuning.hidden_size = tune.qrandint(50, 1000, 25)
        self.tuning.lambda_reg = tune.uniform(0, 1.50)
        self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])

class desn_default(esn_base):
    def __init__(self):
        super().__init__()

    def base_modify(self):
        self.import_path = 'models/stochastic/esn/DeepESN.py'
        self.class_name = 'Deep_ESN'

    def hyper_modify(self):
        self.hyper.num_layers = 8
        self.hyper.hidden_size = 100
        self.hyper.nonlinearity = 'tanh'
        
    def tuning_modify(self):
        self.tuning.num_layers = tune.qrandint(2, 16, 2)
        self.tuning.hidden_size = tune.qrandint(5, 80, 5)
        self.tuning.weight_scaling = tune.uniform(0.6, 0.99)   
        self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])
        
class gesn_default(esn_base):
    def __init__(self):
        super().__init__()

    def base_modify(self):
        self.import_path = 'models/stochastic/esn/GrowingESN.py'
        self.class_name = 'Growing_ESN'

    def hyper_modify(self):
        self.hyper.leaky_r = 1
        self.hyper.hidden_size = 100
        self.hyper.branch_size = 8
        self.hyper.weight_scaling = 1
        self.hyper.hw_bound = (0.66, 0.99)
        self.hyper.nonlinearity = 'sigmoid'

    def tuning_modify(self):
        self.tuning.hidden_size = tune.qrandint(5, 80, 5)
        self.tuning.branch_size = tune.qrandint(4, 30, 2)
        self.tuning.weight_scaling = tune.uniform(0.6, 0.99)
        self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])
        
class iesn_default(gesn_default):
    def __init__(self):
        super().__init__()

    def base_modify(self,):
        self.import_path = 'models/stochastic/esn/GrowingESN.py'
        self.class_name = 'Incremental_ESN'

    def ablation_modify(self):
        self.hyper.branch_size = 100
        self.tuning.dict.pop('branch_size') 
        

class cnn_base(nn_base):
    def __init__(self):
        # self.hyper = Opt()
        # self.tuning = Opt()
        super().__init__()
        self.training = False
        self.arch = 'cnn'

    def hyper_init(self):        
        self.hyper.channel_size = 100


class esm_default(cnn_base):
    def __init__(self):
        super().__init__()

    def base_modify(self):
        self.import_path = 'models/stochastic/cnn/ESM_CNN.py'
        self.class_name = 'ESM_CNN'

    def hyper_init(self):
        self.hyper.channel_size = 100
        self.hyper.candidate_size = 30
        self.hyper.hw_lambda = 0.5
        self.hyper.p_size = 3
        self.hyper.search = 'greedy'
        self.hyper.tolerance = 0
        self.hyper.nonlinearity = 'sigmoid'
        
    def tuning_modify(self):
        self.tuning.hw_lambda = tune.uniform(0.1, 0.99)
        self.tuning.p_size = tune.qrandint(2, 4, 1)
        # self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])

class es_default(esm_default):
    def __init__(self):
        super().__init__()

    def base_modify(self):
        self.import_path = 'models/stochastic/cnn/ESM_CNN.py'
        self.class_name = 'ES_CNN'
    
    def tuning_modify(self):
        self.tuning.kernel_size = tune.qrandint(2, 6, 10)
        self.tuning.hw_lambda = tune.uniform(0.1, 0.99)
        self.tuning.p_size = tune.qrandint(2, 4, 1)
        self.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])


class ice_base(nn_base):
    def __init__(self):
        # self.hyper = Opt()
        # self.tuning = Opt()
        super().__init__()
        self.training = False
        self.arch = 'cnn'
        self.innerTuning = True

    def hyper_init(self):
        self.hyper.patience_bos = 20
        self.hyper.max_cells = 30
        
        self.hyper.esn = nn_base()
        self.hyper.esn.name = 'esn'
        self.hyper.esn.hyper.hidden_size = 50
        self.hyper.esn.hyper.iw_bound = 0.1
        self.hyper.esn.hyper.hw_bound = 1.0
        self.hyper.esn.hyper.weight_scale = 0.9
        self.hyper.esn.hyper.nonlinearity = 'sigmoid'

        
        self.hyper.esn.tuning.hidden_size = tune.qrandint(5, 100, 5)
        self.hyper.esn.tuning.iw_bound = tune.uniform(0.1, 0.99)
        self.hyper.esn.tuning.hw_bound = tune.uniform(0.1, 0.99)
        self.hyper.esn.tuning.fc_io = tune.choice([True])  # attention! this settings should be false when using normal scaler in this models 

        self.hyper.cnn = nn_base()
        self.hyper.cnn.name = 'cnn'
        self.hyper.cnn.hyper.kernel_size = 6
        ## padding need be same for deeper model
        # self.hyper.cnn.hyper.padding = 'same'
        # self.hyper.cnn.hyper.padding_mode = 'circular'
        self.hyper.cnn.hyper.padding = 0
        self.hyper.cnn.hyper.pooling_size = 3
        self.hyper.cnn.hyper.pooling = True
        self.hyper.cnn.hyper.hw_bound = 0.5
        self.hyper.cnn.hyper.nonlinearity = 'sigmoid'


        self.hyper.cnn.tuning.kernel_size = tune.qrandint(2, 10, 1)
        self.hyper.cnn.tuning.hw_bound = tune.uniform(0.05, 0.99)
        self.hyper.cnn.tuning.pooling_size = tune.qrandint(2, 10, 1)
        self.hyper.cnn.tuning.pooling = tune.choice([True, False])
        
        # self.hyper.cnn.tuning.padding_mode = tune.choice(['zeros', 'reflect', 'replicate','circular']) # allow this search space for deeper models
        self.hyper.cnn.tuning.fc_io = tune.choice([True]) # attention! this settings should be false when using normal scaler in this models 
                
        self.hyper.mix = nn_base()
        self.hyper.mix.hyper.kernel_size = 17
        self.hyper.mix.hyper.padding = 0
        self.hyper.mix.hyper.pooling_size = 3
        self.hyper.mix.hyper.pooling = True
        self.hyper.mix.hyper.cnn_hw_bound = 0.5
        self.hyper.mix.hyper.esn_weight_scale = 0.9
        self.hyper.mix.hyper.esn_hidden_size = 100
        self.hyper.mix.hyper.esn_iw_bound = 0.1
        self.hyper.mix.hyper.esn_hw_bound = 0.5
        # self.hyper.mix.hyper.fc_io = True # this line should not be added in the hyper
        self.hyper.mix.hyper.nonlinearity = 'sigmoid'
        
    def tuning_modify(self):
        
        self.hyper.mix.tuning.kernel_size = self.hyper.cnn.tuning.kernel_size
        self.hyper.mix.tuning.pooling_size = self.hyper.cnn.tuning.pooling_size
        self.hyper.mix.tuning.cnn_hw_bound = self.hyper.cnn.tuning.hw_bound
        self.hyper.mix.tuning.esn_hidden_size = self.hyper.esn.tuning.hidden_size
        self.hyper.mix.tuning.esn_iw_bound = self.hyper.esn.tuning.iw_bound
        self.hyper.mix.tuning.esn_hw_bound = self.hyper.esn.tuning.hw_bound
        self.hyper.mix.tuning.fc_io = tune.choice([True])
        self.hyper.mix.tuning.nonlinearity = tune.choice(['tanh', 'sigmoid', 'relu'])
         
        self.hyper.ces = nn_base()
        self.hyper.ces.name = 'ces'
        self.hyper.ces.update(self.hyper.mix)
        
        self.hyper.esc = nn_base()
        self.hyper.esc.name = 'esc'
        self.hyper.esc.update(self.hyper.mix)
        
class wider_default(ice_base):
    def base_modify(self):
        self.import_path = 'models/stochastic/cnn/ice/wider/arch.py'
        self.class_name = 'ICESN'
    
    # def hyper_modify(self):
    #     self.hyper.patience_bos = 100
    #     self.hyper.max_cells = 150
                        
    #     self.hyper.cnn.hyper.kernel_size = 4
                     
    #     # self.hyper.esn.tuning.fc_io = tune.choice([True])   # default true
    #     self.hyper.esn.tuning.hidden_size = tune.qrandint(5, 50, 5)
    #     self.hyper.cnn.tuning.kernel_size = tune.qrandint(30, 90, 10)
        
    def ablation_modify(self):
        self.hyper.cnn.tuner.iters = 1 # equal to randomly select ones without preTuning
        self.hyper.esn.tuner.iters = 1
        self.hyper.esc.tuner.iters = 1
        self.hyper.ces.tuner.iters = 1
        

class wider_pcr_default(wider_default):
    def ablation_modify(self):
        # for pre-tuning
        self.hyper.cnn.tuner.iters = 16 
        self.hyper.esn.tuner.iters = 16
        self.hyper.esc.tuner.iters = 16
        self.hyper.ces.tuner.iters = 16
        #for cell-training
        self.hyper.cTrain = True
        self.hyper.cTrain_info = Opt()
        self.hyper.cTrain_info.training_name = 'naive'
        self.hyper.cTrain_info.max_epoch = 200
        # for readout-tuning
        self.hyper.rTune = True
        self.hyper.ho = Opt()
        self.hyper.ho.tuning = Opt()
        self.hyper.ho.tuning.lambda_reg = tune.uniform(0, 10)
        self.hyper.ho.tuner = Opt()
        self.hyper.ho.tuner.iters = 20
