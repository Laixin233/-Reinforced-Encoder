from ast import arg
import os
from pyexpat import model
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from task.parser import get_parser
from task.TaskWrapper import Task 
from task.TaskRerun import TaskRe
from task.TaskLoader import Opt
import torch
import gc
# from models.stochastic.cnn import ESM_CNN


if __name__ == "__main__":             
    
    args = get_parser(parsing=False)
    
    args.add_argument('-sid',type= int, default=0, help='experimental series id' )
    args.add_argument('-cid', type= int, nargs='+', default=[0], help='experimental cross validation id')
    args.add_argument('-re' ,default=False,action='store_true', help='experimental Rerun')

    args = args.parse_args()
    
    args.test = True
    args.cuda = True
    
    args.datafolder = 'paper/seq'  

    
    # args.tuner_name = 'Random_Search' #['Random_Search','Hyperopt_Search','Ax_Search']
    
    # args.re = False
    # args.re = True     
    # args.cid = [3,9] 
    # gate_opt = Opt()
    # gate_opt.max = 5
    # gate_opt.best = {}
    
    # args.sid = 1
    # args.cid = 5 / 'all'

    args.save_model = True

    args.tune = False
    
    args.tuner_name ='Hyperopt_Search'#['Random_Search','Hyperopt_Search','Ax_Search']

    # data_list =  ['ETTh1','ETTh2','SML1','SML1','mg','ili','ili','Traffic','Electricity','Weather']
    # model_list = ['RNNDQN','RNNPG','RNNPPO','RNNRAN','Naive01','Naive11','RNNPSO','DilatedRNN','PGLSTM','LSTMJUMP','EBPSO','PsLSTM']


    model_list = ['RNNPPO'] #'Naive01','Naive11','RNNRAN','RNNPG','RNNPPO','RNNDQN','DilatedRNN','PGLSTM'
    data_list =  ['Electricity'] #'ETTh1','ili','Traffic','Electricity','Weather','ETTh2','SML1','mg'
    H_list = [24] #12,24,48,96


    args.gid = 1
    
    
    args.rep_times = 1 # 多次求平均和方差 10 15
    
    try:
        for k in range(len(H_list)):
            for i in range(len(model_list)):
                for j in range(len(data_list)):
                    args.model = model_list[i]
                    args.dataset = data_list[j]
                    args.H = H_list[k]
                    
                    if args.re:
                        task = TaskRe(args)
                        # task.selection(gate_opt)
                    else:
                        task = Task(args)
                        task.conduct()
                    # task.tuning()
                    # task.conduct()
                    # # args.metrics = ['rmse']
                    args.metrics = ['rmse','nrmse', 'mase','mape','smape','mse','mae']
                    task.evaluation(args.metrics)
    except Exception as e:
        print(e)
        torch.cuda.empty_cache() if args.cuda else gc.collect()
