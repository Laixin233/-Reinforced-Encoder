# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
# from numpy.lib.function_base import select
from ray import tune
from task.TaskLoader import Opt, TaskDataset
from data.base import esn_base, cnn_base, nn_base, ice_base,seq2seq_base
import numpy as np


class laser_data(TaskDataset):
    def __init__(self, opts):
        super().__init__(opts)

    def info_config(self):
        self.info.normal = False
        self.info.num_series = 1
        self.info.cov_dim = 0
        self.info.steps = 180
        self.info.input_dim = 1
        self.info.period = 60
        self.info.batch_size = 4096

    def sub_config(self,):
        self.seriesPack = []

        for i in range(self.info.num_series):
            raw_ts = np.load(
                'data/paper/seq/laser/laser.npy').reshape(-1,)
            sub = self.pack_dataset(raw_ts)
            sub.index = i
            sub.name = 'laser'
            sub.H = self.info.H
            sub.merge(self.info)
            
            self.seriesPack.append(sub)
