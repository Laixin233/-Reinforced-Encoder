# from sklearn.model_selection import TimeSeriesSplit
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from pandas import Series
from torch.utils.data import Dataset, Sampler
import numpy as np
import torch
import torch.nn as nn

#差分
def difference(dataset, interval=1):
    diff = list()
    diff.append(0)
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff).values

#逆差分
def inverse_diff(opts, pred, raw_test_data):
    _pred = de_scale(opts, pred)
    raw_test_target = raw_test_data[:, (0-opts.H):].reshape(-1, opts.H)
    raw_test_base = raw_test_data[:, (0-opts.H-1):-1].reshape(-1, opts.H)
    raw_test_pred = _pred + raw_test_base

    return raw_test_target, raw_test_pred


def unpadding(y):
    a = y.copy()
    h = y.shape[1]
    s = np.empty(y.shape[0] + y.shape[1] - 1)

    for i in range(s.shape[0]):
        s[i] = np.diagonal(np.flip(a, 1), offset=-i + h-1,
                           axis1=0, axis2=1).copy().mean()

    return s

# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1,horizon=1):
    # dataset = np.insert(dataset, [0] * look_back, 0)
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - horizon + 1):
        x_start = i
        x_end = x_start+ look_back
        y_start = x_end
        y_end = y_start + horizon
        a = dataset[x_start:x_end]
        dataX.append(a)
        if dataset.shape[1] == 1:
            dataY.append(dataset[y_start:y_end])
        else:
            dataY.append(dataset[y_start:y_end,-1])
    dataY = np.array(dataY)
    # dataY = np.reshape(dataY, (dataY.shape[0], 1))
    dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset


def create_patch_dataset(dataset, look_back=1, patch_size=None):
    return




def create_2d_dataset(dataset, T=3, H=1):
    look_back = T + H-1
    datax,datay = [],[]
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        datax.append(a)
        b = dataset[i+look_back]
        datay.append(dataset[i+look_back])
    datay = np.array(datay)
    datay = np.reshape(datay,(datay.shape[0],1))
    dataset = np.concatenate((datax,datay),axis=1)
    return



class scaled_Dataset(Dataset):
    '''
    Packing the input x_data and label_data to torch.dataset
    '''
    def __init__(self, x_data, label_data):
        self.data = np.float32(x_data.copy())
        self.label = np.float32(label_data.copy())
        self.samples = self.data.shape[0]
        # logger.info(f'samples: {self.samples}')

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        return (self.data[index], self.label[index])


def deepAR_dataset(data, train=True, h=None, steps=None, sample_dense=True):
    assert h != None and steps != None
    raw_data = unpadding(data).reshape(-1, 1)
    time_len = raw_data.shape[0]
    input_size = steps
    window_size = h + steps
    stride_size = h
    if not sample_dense:
        windows_per_series = np.full((1), (time_len-input_size) // stride_size)
    else:
        windows_per_series = np.full((1), 1 + time_len-window_size)
    total_windows = np.sum(windows_per_series)

    x_input = np.zeros((total_windows, window_size, 1), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    # v_input= np.zeros((total_windows, 2),dtype = 'float32')

    count = 0
    for i in range(windows_per_series[0]):
        # get the sample with minimal time period, in this case. which is 24 points (24h, 1 day)
        stride = 1
        if not sample_dense:
            stride = stride_size

        window_start = stride*i
        window_end = window_start+window_size
        '''
        print("x: ", x_input[count, 1:, 0].shape)
        print("window start: ", window_start)
        print("window end: ", window_end)
        print("data: ", data.shape)
        print("d: ", data[window_start:window_end-1, series].shape)
        '''
        # using the observed value in the t-1 step to forecast the t step, thus the first observed value in the input should be t0 step and is 0, as well as the first value in the labels should be t1 step.

        x_input[count, 1:, 0] = raw_data[window_start:window_end-1, 0]
        label[count, :] = raw_data[window_start:window_end, 0]

        count += 1

    packed_dataset = scaled_Dataset(x_data=x_input, label_data=label)
    return packed_dataset, x_input, label


def deepAR_weight(x_batch, steps):
    # x_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
    batch_size = x_batch.shape[0]
    v_input = np.zeros((batch_size, 2), dtype='float32')
    for i in range(batch_size):
        nonzero_sum = (x_batch[i, 1:steps, 0] != 0).sum()
        if nonzero_sum.item() == 0:
            v_input[i, 0] = 0
        else:
            v_input[i, 0] = np.true_divide(
                x_batch[i, 1:steps, 0].sum(), nonzero_sum)+1
            x_batch[i, :, 0] = x_batch[i, :, 0] / v_input[i, 0]

    return x_batch, v_input


class deepAR_WeightedSampler(Sampler):
    def __init__(self, v_input, replacement=True):
        v = v_input.copy()
        self.weights = torch.as_tensor(
            np.abs(v[:, 0])/np.sum(np.abs(v[:, 0])), dtype=torch.double)
        # logger.info(f'weights: {self.weights}')
        self.num_samples = self.weights.shape[0]
        # logger.info(f'num samples: {self.num_samples}')
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples


# def de_scale(opts, pred):
#     _pred = pred.copy()
#     ones = np.ones((_pred.shape[0], opts.steps))
#     cat = np.concatenate((ones, _pred), axis=1)
#     _pred = opts.scaler.inverse_transform(cat)[:, -opts.H:]
#     return _pred

# def de_scale(opts, input, tag='target'):
#     assert tag in ['input', 'target']
#     _input = input.copy()
#     if tag == 'input':
#         ones = np.ones((_input.shape[0], opts.H))
#         cat = np.concatenate((_input, ones), axis=1)
#         _input = opts.scaler.inverse_transform(cat)[:, :opts.steps]
#     if tag == 'target':
#         L = opts.steps-opts.patch_size+1
#         ones = np.ones((_input.shape[0], _input.shape[0]))
#         cat = np.concatenate((ones, _input), axis=1)
#         _input = opts.scaler.inverse_transform(cat)[:, -L:]
#     return _input

def de_scale(opts, input, tag='target'):
    """
    逆标准化函数（适配多变量输入，预测1个变量）
    
    Args:
        opts: 包含scaler和配置信息的对象
        input: 需要逆标准化的数据
        tag: 'input' 表示多变量输入(7维), 'target' 表示单变量输出(1维)
    
    Returns:
        逆标准化后的数据
    """
    assert tag in ['input', 'target']
    
    if input is None:
        return None
    
    _input = input.copy()
    
    if tag == 'input':
        # 多变量输入 (batch, seq_len, 7)
        if _input.ndim == 3:
            batch_size, seq_len, num_features = _input.shape
            
            # Reshape to (batch*seq_len, 7)
            _input_reshaped = _input.reshape(-1, num_features)
            # 逆标准化
            _input_inversed = opts.scaler.inverse_transform(_input_reshaped)
            # Reshape back
            return _input_inversed.reshape(batch_size, seq_len, num_features)
        elif _input.ndim == 2:
            # (batch, 7)
            return opts.scaler.inverse_transform(_input)
        else:
            raise ValueError(f"Unexpected input shape: {_input.shape}")
    
    elif tag == 'target':
        # 单变量输出 (batch, H) 或 (batch, H, 1)
        # 获取目标变量（假设是最后一个特征，索引-1）的均值和标准差
        target_mean = opts.scaler.mean_[-1]
        target_scale = opts.scaler.scale_[-1]
        
        # 逆标准化: x_original = x_scaled * scale + mean
        _input_inversed = _input * target_scale + target_mean
        
        return _input_inversed
    
    return _input


def re_scale(opts, input, tag=None):
    assert tag in ['input', 'target']
    _input = input.copy()
    if tag == 'input':
        ones = np.ones((_input.shape[0], opts.H))
        cat = np.concatenate((_input, ones), axis=1)
        _input = opts.scaler.transform(cat)[:, :opts.steps]
    if tag == 'target':
        ones = np.ones((_input.shape[0], opts.steps))
        cat = np.concatenate((ones, _input), axis=1)
        _input = opts.scaler.transform(cat)[:, -opts.H:]
    return _input




def mlp_dataset(data, h, steps):
    x = data[:, :(0 - h)].reshape(data.shape[0], steps)
    y = data[:, (0-h):].reshape(-1, h)
    data_set = scaled_Dataset(x_data=x, label_data=y)

    return data_set, x, y

def rnn_dataset(data, h, steps,otherVariate_list = None):
    '''
    x, shape: (N_samples, steps, dimensions(input_dim/multivariate))\n
    y, shape: (N_samples, dimensions)
    '''
    if otherVariate_list is not None:
        list = []
        for var in range(len(otherVariate_list)):
            data_var = otherVariate_list[var]
            list.append(data_var[:, :(0 - h)].reshape(data_var.shape[0], steps , 1))
        data_var_else = np.concatenate(list,axis=2)
        x = np.concatenate((data_var_else,data[:, :(0 - h)].reshape(data.shape[0], steps , 1)),axis=2)
    else:
        x = data[:, :(0 - h)].reshape(data.shape[0], steps , 1)
    y = data[:, (0-h):].reshape(-1, h)


    data_set = scaled_Dataset(x_data=x, label_data=y)

    return data_set, x, y

#折叠数据
def unfolded(x,d):
    B,T = x.shape[0],x.shape[1]
    L = T - d + 1
    unfold = []
    for i in range(L):
        window = x[:,i:i+d]
        unfold.append(window)
    unfolds = np.stack(unfold, axis=1)
    return unfolds

def rnn_patch_dataset(data, h, steps,patch_size=None,otherVariate_list = None):
    '''
    x, shape: (N_samples, steps, dimensions(input_dim))\n
    y, shape: (N_samples, dimensions)
    '''
    if otherVariate_list is not None:
        list = []
        for var in range(len(otherVariate_list)):
            data_var = otherVariate_list[var]
            list.append(data_var[:, :(0 - h)].reshape(data_var.shape[0], steps , 1))
        data_var_else = np.concatenate(list,axis=2)
        x = np.concatenate((data_var_else,data[:, :(0 - h)].reshape(data.shape[0], steps , 1)),axis=2)
    else:
        x = data[:, :(0 - h)].reshape(data.shape[0], steps)
    y = data[:, patch_size:].reshape(data.shape[0], -1)
    L = steps-patch_size+1
    if patch_size is not None:
        datax = unfolded(x,patch_size)
        datay = unfolded(y,h)
        # y = y.reshape(y.shape[0],L,-1)


    data_set = scaled_Dataset(x_data=datax, label_data=datay)

    return data_set, x, y



def dnn_dataset(data, h, steps):
    '''
    x, shape: (N_samples, dimensions(input_dim), steps)\n
    y, shape: (N_samples, dimensions)
    '''
    x = data[:, :(0 - h)].reshape(data.shape[0], 1,steps)
    y = data[:, (0-h):].reshape(-1, h)
    data_set = scaled_Dataset(x_data=x, label_data=y)

    return data_set, x, y


# def rnn_dataset(data, h, steps, expand_dim=1):
#     y = data[:, (0-h):].reshape(-1, h)

#     x = np.zeros((data.shape[0], steps, expand_dim))
#     for i in range(expand_dim):
#         x[:, :, i] = data[:, i:steps+i]
#     # return the X with shape: samples, timesteps, dims

#     data_set = scaled_Dataset(x_data=x, label_data=y)

#     return data_set, x, y

class Dataset_normal(Dataset):
    def __init__(self, data, info=None):
        self.data = data
        self.data_x = data
        self.data_y = data[:, -1]
        self.seq_len = info.steps
        self.pred_len = info.H
 
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        if isinstance(seq_x, np.ndarray):
            seq_x = torch.from_numpy(seq_x)
        if isinstance(seq_y, np.ndarray):
            seq_y = torch.from_numpy(seq_y)

        return (seq_x, seq_y)#x:t,c y:h,1(t,h)

    def get_last_data(self):
        start_index = max(0, len(self.data_x) - self.seq_len)
        x_last = self.data_x[start_index:]
        x_mark = self.data_stamp[start_index:]
        return x_last,x_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_nonpatch(Dataset):
    def __init__(self, data, info=None):
        super().__init__()
        self.data = data
        self.data_x = data
        self.data_y = data[:, -1]
        self.seq_len = info.steps
        self.pred_len = info.H 

 
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = index + 1
        r_end = r_begin + self.seq_len +self.pred_len - 1

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]


        if isinstance(seq_x, np.ndarray):
            Data_x = torch.from_numpy(seq_x).float()
        if isinstance(seq_y, np.ndarray):
            seq_y = torch.from_numpy(seq_y).float()
        
        # data_x = seq_x.unfold(dimension=0, size=self.kernel, step=self.stride).permute(0,2,1)#t,p,c
        Data_y = seq_y.unfold(dimension=0, size=self.pred_len, step=1).squeeze(1)
 

        return (Data_x,Data_y) #data_x:tc, data_y:th

    def get_last_data(self):
        start_index = max(0, len(self.data_x) - self.seq_len)
        x_last = self.data_x[start_index:]
        x_mark = self.data_stamp[start_index:]
        return x_last,x_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len -1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_patch(Dataset):
    def __init__(self, data, info=None):
        super().__init__()
        self.data = data
        self.data_x = data
        self.data_y = data[:, -1]
        self.seq_len = info.steps
        self.pred_len = info.H 
        self.kernel = info.patch_size
        self.stride = info.stride
 
    def __getitem__(self, index):
       
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y #[r_begin:r_end]

        if isinstance(seq_x, np.ndarray):
            seq_x = torch.from_numpy(seq_x).float()
        if isinstance(seq_y, np.ndarray):
            seq_y = torch.from_numpy(seq_y).float()
        
        data_x = seq_x.unfold(dimension=0, size=self.kernel, step=self.stride).permute(0,2,1)#t,p,c
        # data_y = seq_y.unfold(dimension=0, size=self.kernel, step=self.stride)#
        data_y = []
        for i in range(data_x.size(0)):
            start_idx = s_begin + i * self.stride + self.kernel 
            end_idx = start_idx + self.pred_len 
            y = seq_y[start_idx:end_idx]

            # # 检查边界
            # if end_idx > len(self.data_y):
            #     raise IndexError(f"索引 {index} 的 i={i} 超出范围: end_idx={end_idx} 超过数据长度 {len(self.data_y)}")

            data_y.append(y)
        

        if len(data_y[26]) == 23:
            raise IndexError("Index out of range")
        # if index >self.__len__():
        #     raise IndexError("Index out of range")


        Data_y = torch.stack(data_y, dim=0)

        Data_x = data_x.reshape(data_x.size(0), -1)
        

        return (Data_x,Data_y) #data_x:(l,pc), data_y:(l,h)

    def get_last_data(self):
        start_index = max(0, len(self.data_x) - self.seq_len)
        x_last = self.data_x[start_index:]
        x_mark = self.data_stamp[start_index:]
        return x_last,x_mark

    def __len__(self):
        # # 计算允许的最大s_begin，确保所有分块的end_idx不超过数据长度

        return len(self.data_x) - self.seq_len - self.pred_len -5

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

        # num_patches = (self.seq_len - self.kernel) // self.stride + 1
        # max_s_begin = len(self.data_x) - (self.seq_len + (num_patches - 1) * self.stride + self.kernel + self.pred_len)
        # return max_s_begin + 1  # 返回合法索引的数量