'''
Attention !
************
For stochastic model, e.g. ESN, DESN,..., etc. 
They are trained only once that are solving their output-weight in a close form manner. Thus the schedulers based Tuner cannot be implemented into tuning these models, that the tuner will sample the hypers from the config (tuning.dict) only once, and will not tuning sequentially further, causing the parameters 'tuner.iters' is meaningless. 
'''

import os
import sys
from numpy import not_equal

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
# import json
import ray
from task.TaskLoader import Opt
# from ray.tune.suggest.bohb import TuneBOHB
# from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.schedulers import PopulationBasedTraining 
# https://arxiv.org/pdf/1711.09846.pdf.
from ray.tune.schedulers.pb2 import PB2 
# pip install GPy sklearn
# https://arxiv.org/abs/2002.02518 (NIPS 2020)
from ray import tune,air
from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search.ax import AxSearch
from ray.tune.search import BasicVariantGenerator
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandScheduler
from ray.air import session


import nevergrad as ng

import importlib
import torch

import pandas as pd

import logging


class Tuner(Opt):
    def __init__(self, opts = None):
        super().__init__()

        if opts is not None:
            self.merge(opts)

        self.best_config = None
        
    def conduct(self,):
        analysis = self._conduct()
        self.best_result = analysis.get_best_result()
        self.best_config = self.best_result.config
        print("Best config is:", self.best_config)

    def _conduct(self,):
        '''Tuning'''
        pass

    def fitness(self, config):
        '''Loading the hyper-parameters in config, then return a score'''
        pass
    
    # def search_ng(self,):
    #     """
    #     https://github.com/facebookresearch/nevergrad.
    #     """
    #     self.tuner.name = 'PSO_Search'
    #     self.tuner.algo = 'algo'
        
    #     ng_search = NevergradSearch(
    #         optimizer=ng.optimizers.ConfiguredPSO(popsize=20),
    #         metric=self.metric,
    #         mode="min",)
    #     return ng_search
    
    # def search_bohb(self,points_eval):
    #     self.tuner.name = 'BOHB_Search'
    #     self.tuner.algo = 'algo'
    #     bohb_search = TuneBOHB()
    #     return bohb_search
    
    # def search_bayes(self,points_eval):
    #     self.tuner.name = 'Bayes_Search'
    #     self.tuner.algo = 'algo'
    #     byes_search = BayesOptSearch(
    #         # points_to_evaluate=points_eval,
    #         random_search_steps = 10,
    #         random_state = 42,
    #         verbose = 1)
    #     return byes_search

class StocHyperTuner(Tuner):
    def __init__(self, opts, logger, subPack):
        super().__init__(opts)
        self.logger = logger
        self.train_loader = subPack.train_loader
        self.valid_loader = subPack.valid_loader
        # for testing
        self.tuner.metric = 'best_vrmse' # to do: comment
        self.metric = self.tuner.metric
        
        if 'iters' not in self.tuner.dict:
            self.tuner.iters = 20

    def _fitness(self, model):
        '''
        config is equal to self.tuning
        '''
        train_loader = self.train_loader
        valid_loader = self.valid_loader

        fit_info = model.xfit(train_loader, valid_loader)
        del model
        return fit_info

    def fitness(self, config):
        _hyper = Opt()
        _hyper.merge(self.hyper)
        _hyper.update(config) 
        model = importlib.import_module(self.import_path)
        model = getattr(model, self.class_name)         
        model = model(_hyper, self.logger)

        best_trmse = float('inf')
        best_vrmse = float('inf')
        
        for epoch in range(self.tuner.epochPerIter):
            fit_info = self._fitness(model)
            trmse, vrmse = fit_info.trmse, fit_info.vrmse
            
            if trmse <= best_trmse:
                best_trmse = trmse
            if vrmse <= best_vrmse:
                best_vrmse = vrmse
            best_vrmse = best_vrmse.item() if isinstance(best_vrmse, torch.Tensor) else best_vrmse
            best_trmse = best_trmse.item() if isinstance(best_trmse, torch.Tensor) else best_trmse
            trmse = trmse.item() if isinstance(trmse, torch.Tensor) else trmse
            vrmse = vrmse.item() if isinstance(vrmse, torch.Tensor) else vrmse    
            session.report({'best_trmse':best_trmse,
                            'cur_trmse':trmse,
                            'best_vrmse' : best_vrmse,
                            'cur_vrmse': vrmse})
    
    def search_random(self,points_eval):
        self.tuner.name = 'Random_Search'
        self.tuner.algo = 'algo'
        basic_search = BasicVariantGenerator(
            points_to_evaluate= points_eval,
            random_state = 10)
        return basic_search
    
    def search_ax(self,points_eval):
        self.tuner.name = 'Ax_Search'
        self.tuner.algo = 'algo'
        ax_search = AxSearch(
            points_to_evaluate= points_eval
        )
        return ax_search
    
    def search_hyperopt(self,points_eval):
        self.tuner.name = 'Hyperopt_Search'
        self.tuner.algo = 'algo'
        hyperopt_search = HyperOptSearch(
            points_to_evaluate = points_eval,
            random_state_seed = 40
        )
        return hyperopt_search    
    
    def algo_run(self, algo):
        _tune = tune.Tuner(
            tune.with_resources(
                self.fitness,
                resources = self.resource
            ),
            param_space = self.tuning.dict,
            tune_config = tune.TuneConfig(
                metric=self.metric,
                mode="min",
                search_alg=algo,
                num_samples=self.tuner.iters,    
                max_concurrent_trials = 2
            ),
            run_config = air.RunConfig(
                name = self.tuner.name,
                #local_dir = self.tuner.dir,
                storage_path = self.tuner.dir,
                verbose = 1
            )
        )
        analysis = _tune.fit()
        return analysis

    
    def save(self, analysis):
        all_results = analysis

        results = pd.DataFrame()
        for i in range(len(all_results)):
            result = all_results[i]
            df = pd.DataFrame()
            record = {}
            for key in ['trial_id', 'training_iteration','best_trmse','cur_trmse','best_vrmse','cur_vrmse']:
                if key in result.metrics:
                    record[key] = result.metrics[key]
                else:
                    record[key] = 'error'
            df = pd.DataFrame(record, index=[0])

            _df = pd.DataFrame(result.config,index=[0])
            df = pd.concat([df, _df],axis= 1)
            results = pd.concat([results, df]).reset_index(drop=True)
        results.to_csv(os.path.join(self.tuner.dir, '{}.trial.csv'.format(self.tuner.name)))
    
    def _conduct(self):
        # ray.init number of cpu,gpu available
        if self.hyper.device.type == 'cuda':
            ray.init(num_cpus = self.tuner.cores,num_gpus = self.tuner.cards)
            self.resource = {
                "cpu": self.tuner.cores,
                "gpu": self.tuner.cards  # set this for GPUs
            }
        else:
            ray.init()
            self.resource = {
                "cpu": self.tuner.cores
            }
        # loading Initial parameter suggestions to be run first
        if self.tuner.points_eval is None:
            tuner_first = {}
            for key in self.tuning.dict:
                tuner_first[key] = self.hyper.dict[key]
            points_eval = [tuner_first]
        else:
            points_eval = self.tuner.points_eval
        
        # loading search_method
        if 'name' not in self.tuner.dict:
            self.tuner.name = 'Random_Search'
            algo = self.search_random(points_eval) 
        else:
            if self.tuner.name  == 'Hyperopt_Search':
                # ok
                algo = self.search_hyperopt(points_eval)
            elif self.tuner.name  == 'Ax_Search':
                algo = self.search_ax(points_eval)
            else:
                algo = self.search_random(points_eval)
        analysis = self.algo_run(algo)
        self.save(analysis)
        ray.shutdown()
        return analysis