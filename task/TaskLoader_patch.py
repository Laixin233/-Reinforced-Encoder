import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler



class Dataset_RNN_RL(Dataset):
    def __init__(self, patch_len, stride, padding, root_path, flag='train', size=None,
                 features='S', data_path='data/paper/seq/real/ili/ILI.csv',
                 target='north_ILI', scale=True, timeenc=0):
        self.seq_len = size[0]
        self.pred_len = size[1]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path

        self.d = patch_len
        self.stride = stride
        self.padding = padding
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        self.__read_data__()
        
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        print(f"Raw data shape: {df_raw.shape}")  # 打印原始数据形状

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('time')
        df_raw = df_raw[['time'] + cols + [self.target]]

        print(f"Processed data columns: {df_raw.columns}")  # 打印处理后的数据列

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        print(f"Border1: {border1}, Border2: {border2}")  # 打印边界索引

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        print(f"Data shape after scaling: {data.shape}")  # 打印缩放后的数据形状
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        print(f"Data_x shape: {self.data_x.shape}, Data_y shape: {self.data_y.shape}")  # 打印数据形状
    
    def unfolded(self, x):
        T,C = x.shape[0],x.shape[1]
        L = T - self.d + 1
        unfold = []
        for i in range(L):
            window = x[i:i+self.d, :]
            unfold.append(window)
        unfolds = np.stack(unfold, axis=0)
        return unfolds
    
    def __getitem__(self, index):
        #seq_x(T,C)
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = self.d + index
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_ = self.unfolded(seq_x)
        seq_y_ = self.unfolded(seq_y)
        return seq_x_, seq_y_
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scale.inverse_transform(data)

train_dataset = Dataset_RNN_RL(
    patch_len=5,
    stride=1,
    padding=2,
    root_path='./data/paper/seq/real/ili/',
    flag='train',
    size=[12, 8],
    features='MS',
    data_path='ILI.csv',
    target='north_ILI',
    scale=True,
    timeenc=0
    )
test_dataset = Dataset_RNN_RL(
    patch_len=5,
    stride=1,
    padding=2,
    root_path='./data/paper/seq/real/ili/',
    flag='test',
    size=[12, 8],
    features='MS',
    data_path='ILI.csv',
    target='north_ILI',
    scale=True,
    timeenc=0
)
train_loader = DataLoader(
    train_dataset,
    batch_size=30,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True
)
for batch_x, batch_y in train_loader:
    print(batch_x.shape)




