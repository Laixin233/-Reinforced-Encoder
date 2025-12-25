import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from tqdm.std import tqdm
import numpy as np

from tqdm import trange

from task.util import os_makedirs, os_rmdirs, set_logger
from task.util import plot_xfit
import shutil
from task.TaskWrapper import Task
import logging
import random

class TaskRe(Task):
    def __init__(self, args):
        super().__init__(args)
        self.sid = args.sid
        self.cid = args.cid

                
    def conduct(self,):
        # init and mkdir taskdir
        # generate the subPack dataset            
        if self.tune:
            self.tuning()
            
        # for sub_count, series_Pack in enumerate(tqdm(self.data_opts.seriesPack)):
            
        sub_count = self.sid
        series_Pack = self.data_opts.seriesPack[sub_count]
        # self.task_dir = os.path.join(self.task_dir, 'series_{}'.format(sub_count))
        assert sub_count == series_Pack.index
        self.series_dir = os.path.join(self.task_dir, 'series{}'.format(sub_count))
        self.measure_dir = os.path.join(self.series_dir, 'eval_results')
        os_makedirs(self.measure_dir)

        # for i in trange(self.rep_times):

        # i = self.cid
        for i in self.cid:
            result_file = os.path.join(
                self.measure_dir, 'results_{}.series_{}.npy'.format(i, sub_count))

            # if os.path.exists(result_file):
            #     continue
            if i > 0 and 'statistic' in self.model_opts.dict:
                raise ValueError(
                    'Task-Rerun function only support non-statistic models')
                
                # assert self.model_opts.statistic
                # result0 = str(os.path.join(
                #     self.measure_dir, 'results_{}.series_{}.npy'.format(0, sub_count)))
                # shutil.copy(result0, result_file)
                

            # loading the best paramters:
            cLogger = self.logger_config(self.series_dir,'train',i,sub_count)
            cLogger.critical('*'*80)
            cLogger.critical('Dataset: {}\t Model:{} \t H: {}\t Trail: {}'.format(
                self.data_name, self.model_name, self.model_opts.hyper.H, i))
                
            self.conduct_iter(i, series_Pack, result_file, cLogger,_seed = i + self.rep_times)
    
    def selection(self, gate_opt):
        sub_count = self.sid
        series_Pack = self.data_opts.seriesPack[sub_count]
        # self.task_dir = os.path.join(self.task_dir, 'series_{}'.format(sub_count))
        assert sub_count == series_Pack.index
        self.series_dir = os.path.join(self.task_dir, 'series{}'.format(sub_count))
        self.measure_dir = os.path.join(self.series_dir, 'eval_results')
        os_makedirs(self.measure_dir)

        # for i in trange(self.rep_times):
        self.metrics = gate_opt.metrics.keys()
            
        # i = self.cid
        if self.cid == ['all']:
            self.cid = list(range(self.rep_times))
        
        for i in self.cid:
            i = int(i)
            reTag = True
            
            result_file = os.path.join(
                self.measure_dir, 'results_{}.series_{}.npy'.format(i, sub_count))
            metric_file = os.path.join(
                    self.measure_dir, 'metrics_{}.series_{}.npy'.format(i, sub_count))
            
            if i > 0 and 'statistic' in self.model_opts.dict:
                raise ValueError(
                    'Task-Rerun function only support non-statistic models')
            reTimes = 0
            for _, eval_name in enumerate(self.metrics):
                gate_opt.best[eval_name] = 9999
            while reTag:
            # loading the best paramters:
                eval_results = self.eval_iter(
                                i, sub_count)
                logging.critical('*'*80)
                logging.critical('Dataset: {}\t Model: {} \t H: {} \t Series-id: {} \t Trail-id: {} \t Re-times: {}'.format(
                    self.data_name, self.model_name, self.data_opts.info.H,sub_count, i, reTimes))  # to do: changing to sub.H for diversing H
                
                for _i, eval_name in enumerate(self.metrics):
                    _error = eval_results[0, _i]
                    logging.critical(
                        'Testing\t{}:\t{:.4g}'.format(eval_name, _error))
                    
                    if _error <= gate_opt.best[eval_name]:
                        gate_opt.best[eval_name] = _error                    
                    logging.critical('Current Best {}: {}'.format(eval_name, gate_opt.best[eval_name]))
                    
                    if _error <= gate_opt.metrics[eval_name]:
                        reTag = False
                    else:
                        reTag = True
                        break
                
                if reTag == False:
                    np.save(metric_file, eval_results)
                    break
                else:
                    cLogger = self.logger_config(self.series_dir,'train',i,sub_count)
                    cLogger.critical('*'*80)
                    # cLogger.critical('Dataset: {}\t Model:{} \t H: {}\t Trail: {}'.format(
                    # self.data_name, self.model_name, self.model_opts.hyper.H, i, reTimesz))
                
                    reTimes += 1
                    self.conduct_iter(i, series_Pack, result_file, cLogger,random.randint(0,500) + reTimes)
            
                if reTimes >= gate_opt.max:
                    break
                
            np.save(metric_file, eval_results)
            
    def logger_config(self, dir, stage, cv, sub_count):
        log_path = os.path.join(dir, 'logs',
                                '{}.cv{}.series{}.log'.format(stage, cv, sub_count))
        log_name = '{}.series{}.cv{}.{}'.format(
            self.data_name, sub_count, cv, self.model_name)
        logger = set_logger(log_path, log_name, self.logger_level)
        return logger   
    
          
    def evaluation(self,  metrics=['rmse']):
        try:
            self.metrics = metrics
            # eval_list = []
            # logging = set_logger(os.path.join(self.task_dir, 'eval.log'),'{}.H{}.{}'.format(self.data_name, self.data_opts.info.H, self.model_name.upper()),self.logger_level)
            
            # for sub_count in range(self.data_opts.info.num_series):
            sub_count = self.sid
            self.series_dir = os.path.join(self.task_dir, 'series{}'.format(sub_count))
            self.measure_dir = os.path.join(self.series_dir, 'eval_results')
            os_makedirs(self.measure_dir)
            
            # for i in range(self.rep_times):
            for i in self.cid:
                metric_file = os.path.join(
                        self.measure_dir, 'metrics_{}.series_{}.npy'.format(i, sub_count))
                
                eval_results = self.eval_iter(
                            i, sub_count)
                # logging = self.logger_config(self.series_dir,'eval',i,sub_count)
                                
                logging.critical('*'*80)
                logging.critical('Dataset: {}\t Model: {} \t H: {} \t Series-id: {} \t Trail-id: {}'.format(
                    self.data_name, self.model_name, self.data_opts.info.H,sub_count, i))  # to do: changing to sub.H for diversing H
                for _i, eval_name in enumerate(self.metrics):
                        logging.critical(
                            'Testing\t{}:\t{:.4g}'.format(eval_name, eval_results[0, _i]))
                # eval_list.append(eval_results)
                logging.critical('-'*80)
                np.save(metric_file, eval_results)
            # eval_data = np.concatenate(eval_list, axis=0)
            
        except:
            logging.exception('{}\nGot an error on evaluation.\n{}'.format('!'*50,'!'*50))
            raise SystemExit()