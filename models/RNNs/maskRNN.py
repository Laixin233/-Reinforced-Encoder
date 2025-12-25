import torch
import torch.nn as nn
from models.RNNs.PyxLSTM.xLSTM import xLSTM
from models.RNNs.PeepholeLSTM.Peepholelstm import PeepholeLSTM
from models.RNNs.indRNN.indrnn import IndRNN
from models.RNNs.PhaseLSTM.phasedLSTM import PhasedLSTM
from models.RNNs.MGU.mgu import MGU


class MaskEncoder(nn.Module):
    def __init__(self, component, input_size, hidden_size, num_layers, out_size, device):
        super().__init__()
        self.component = component
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # 移除重复的layer定义
        if component == 'LSTM':
            self.layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'GRU':
            self.layer = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'MGU':
            self.layer = MGU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'PeepholeLSTM':
            self.layer = PeepholeLSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'indRNN':
            self.layer = IndRNN(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'xLSTM':
            self.layer = xLSTM(input_size, hidden_size, num_layers,num_blocks=1).to(device)
        
        elif component == 'phasedLSTM':
            self.layer = PhasedLSTM(input_size, hidden_size, num_layers,batch_first=True).to(device)
        else:
            self.layer = nn.RNN(input_size, hidden_size, num_layers, batch_first=True).to(device)
            
        self.linear = nn.Linear(hidden_size, out_size).to(device)
        self.component_list = ['LSTM', 'PeepholeLSTM', 'indRNN', 'xLSTM','phasedLSTM']

    def forward(self, input_seq, h=None, c=None, m=None):
        batch_size, seq_len,_ = input_seq.shape
        times = torch.arange(seq_len).to(self.device)

        # mask
        a_x = m[:, 0].unsqueeze(-1) # (batch_size, seq_len)
        a_h = m[:, 1].unsqueeze(-1)  
        a_y = m[:, 2].unsqueeze(-1)  


        # 应用输入mask
        x = input_seq * a_x.unsqueeze(-1)  # (batch_size, seq_len, input_size)
        component_list = ['LSTM', 'PeepholeLSTM', 'indRNN', 'xLSTM','phasedLSTM']
        if self.component in component_list:
            # 初始化隐藏状态
            if h is None:
                h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
            if c is None:
                c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

            # 根据mask更新状态
            mask = a_h.unsqueeze(0).expand(self.num_layers, -1, -1)  # (num_layers, batch_size, seq_len)
            
            if self.component == 'phasedLSTM':
                output, (h_new, c_new) = self.layer(x, (h, c), times)
            else:
                output, (h_new, c_new) = self.layer(x, (h, c)) #b,t,h
            
            # 选择性更新
            h = torch.where(mask == 1, h_new, h)
            c = torch.where(mask == 1, c_new, c)

        else:  # GRU or RNN or MGU
            if h is None:
                h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

            mask = a_h.unsqueeze(0).expand(self.num_layers, -1, -1)
            output, h_new = self.layer(x, h)
            
            # 选择性更新
            h = torch.where(mask == 1, h_new, h)

        # 应用线性层和输出mask
        y = self.linear(output)  # (batch_size, seq_len, out_size)
        y = y * a_y.unsqueeze(-1)  # 应用输出mask

        if self.component in self.component_list:
            return h, c, y
        else:
            return h, None, y

#normal rnn 
class Encoder(nn.Module):
    def __init__(self, component, input_size, hidden_size, num_layers, out_size, device):
        super().__init__()
        self.component = component
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.component_list = ['LSTM', 'PeepholeLSTM', 'indRNN', 'xLSTM','phasedLSTM']

        # 移除重复的layer定义
        if component == 'LSTM':
            self.layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'GRU':
            self.layer = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'MGU':
            self.layer = MGU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'PeepholeLSTM':
            self.layer = PeepholeLSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'indRNN':
            self.layer = IndRNN(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'xLSTM':
            self.layer = xLSTM(input_size, hidden_size, num_layers,num_blocks=1).to(device)
        
        elif component == 'phasedLSTM':
            self.layer = PhasedLSTM(input_size, hidden_size, num_layers,batch_first=True).to(device)
        else:
            self.layer = nn.RNN(input_size, hidden_size, num_layers, batch_first=True).to(device)
            
        self.linear = nn.Linear(hidden_size, out_size).to(device)

    def forward(self, input_seq, h=None, c=None, m=None):
        batch_size, seq_len,_ = input_seq.shape
        times = torch.arange(seq_len).to(self.device)
        
        if h is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        if c is None:
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)


        if self.component in self.component_list:         
            if self.component == 'phasedLSTM':
                output, (h_new, c_new) = self.layer(input_seq, (h, c), times)
            else:
                output, (h_new, c_new) = self.layer(input_seq, (h, c)) #b,t,h            
        else:  # GRU or RNN or MGU

            output, h_new = self.layer(input_seq, h)            

        # 应用线性层和输出mask
        y = self.linear(output)  # (batch_size, seq_len, out_size)
        if self.component in self.component_list:
            return h_new, c, y
        else:
            return h_new, None, y

class CMaskEncoder(nn.Module):
    def __init__(self, component, input_size, hidden_size, num_layers, out_size, device):
        super().__init__()
        self.component = component
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # 移除重复的layer定义
        if component == 'LSTM':
            self.layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'GRU':
            self.layer = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'MGU':
            self.layer = MGU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'PeepholeLSTM':
            self.layer = PeepholeLSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'indRNN':
            self.layer = IndRNN(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'xLSTM':
            self.layer = xLSTM(input_size, hidden_size, num_layers,num_blocks=1).to(device)
        
        elif component == 'phasedLSTM':
            self.layer = PhasedLSTM(input_size, hidden_size, num_layers,batch_first=True).to(device)
        else:
            self.layer = nn.RNN(input_size, hidden_size, num_layers, batch_first=True).to(device)
            
        self.linear = nn.Linear(hidden_size, out_size).to(device)

    def forward(self,input_seq,h=None,c=None,m=None):
        batch_size,seq_len,feature_size = input_seq.shape
        times = torch.arange(feature_size).to(self.device)
        

        # 确保mask维度正确
        a_x = m[0] # (batch_size, seq_len)
        a_h = m[1] # (batch_size, seq_len) .reshape(-1,1)
        a_y = m[2] # (batch_size, seq_len)

        x = input_seq * a_x #(batch_size, 1, input_size)

        component_list = ['LSTM', 'PeepholeLSTM', 'indRNN', 'xLSTM','phasedLSTM']

        # 初始化隐藏状态
        if h is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        if c is None:
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        
            # 根据mask更新状态
        if a_y == 1:
            if self.component in component_list:
                output, (h_new, c_new) = self.layer(x, (h, c)) #b,t,h
            elif self.component == 'phasedLSTM':
                output, (h_new, c_new) = self.layer(x, (h, c), times)
            else:
                output, h_new = self.layer(x, h)
        else:
            h_new = h
            c_new = c
            output = h[-1].unsqueeze(1)


        # 应用线性层和输出mask
        y = self.linear(output)  # (batch_size, seq_len, out_size)
        # y = y * a_y # 应用输出mask,不在这一步mask，放在reward奖励处，限制最后一步一定有输出

        if self.component in component_list:
            return h_new, c_new, y
        else:
            return h_new, None, y

#新的网络结构，包括跳多层。
class MaskJumpEncoder(nn.Module):
    def __init__(self, component, input_size, hidden_size, num_layers, out_size, device):
        super().__init__()
        self.component = component
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        if component == 'LSTM':
            self.layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'GRU':
            self.layer = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'MGU':
            self.layer = MGU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'PeepholeLSTM':
            self.layer = PeepholeLSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'indRNN':
            self.layer = IndRNN(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'xLSTM':
            self.layer = xLSTM(input_size, hidden_size, num_layers, num_blocks=1).to(device)
        elif component == 'phasedLSTM':
            self.layer = PhasedLSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        else:
            self.layer = nn.RNN(input_size, hidden_size, num_layers, batch_first=True).to(device)
        
        # 跳跃连接融合层RNcell(u_t ⊙ x_t, h_{t-1}, h_{t-k_t-1}) 融合 h_{t-1} 和 h_{t-k_t-1}
        self.skip_fusion = nn.Linear(hidden_size * 2, hidden_size).to(device)
        
        self.linear = nn.Linear(hidden_size, out_size).to(device)
        self.component_list = ['LSTM', 'PeepholeLSTM', 'indRNN', 'xLSTM', 'phasedLSTM']
    
    def forward(self, input_seq, h_t=None, c=None, h_skip=None):
        """
        前向传播 - 支持跳跃连接        
        参数:
            input_seq: (batch_size, seq_len, input_size) - 已经应用了 a_x mask
            h_t: 当前隐藏状态 h_{t-1} (num_layers, batch_size, hidden_size)
            c: cell state (num_layers, batch_size, hidden_size)
            h_skip: 跳跃连接的隐藏状态 h_{t-k_t-1} (num_layers, batch_size, hidden_size)        
        返回:
            h_new: 新隐藏状态 h_t
            c_new: 新 cell state
            y: 输出预测
        """
        batch_size, seq_len, _ = input_seq.shape
        times = torch.arange(seq_len).to(self.device)
        
        # 初始化隐藏状态
        if h_t is None:
            h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        if c is None:
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        if h_skip is not None and torch.any(h_skip != 0):
            # 融合当前隐藏状态和跳跃状态
            h_combined = torch.cat([h_t, h_skip], dim=-1)  # (num_layers, batch, 2*hidden)
            h_t_fused = self.skip_fusion(h_combined)  # (num_layers, batch, hidden)
        else:
            # k_t = 0: 不使用跳跃连接
            h_t_fused = h_t

        
        # RNN 前向传播
        if self.component in self.component_list:
            if self.component == 'phasedLSTM':
                output, (h_new, c_new) = self.layer(input_seq, (h_t_fused, c), times)
            else:
                output, (h_new, c_new) = self.layer(input_seq, (h_t_fused, c))
        else:  # GRU or RNN or MGU
            output, h_new = self.layer(input_seq, h_t_fused)
            c_new = c

        y = self.linear(output)  # (batch_size, seq_len, out_size)
        
        if self.component in self.component_list:
            return h_new, c_new, y
        else:
            return h_new, None, y





class MMaskEncoder(nn.Module):
    def __init__(self, component, input_size, hidden_size, num_layers, out_size, device):
        super().__init__()
        self.component = component
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # 移除重复的layer定义
        if component == 'LSTM':
            self.layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'GRU':
            self.layer = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'PeepholeLSTM':
            self.layer = PeepholeLSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'indRNN':
            self.layer = IndRNN(input_size, hidden_size, num_layers, batch_first=True).to(device)
        elif component == 'xLSTM':
            self.layer = xLSTM(input_size, hidden_size, num_layers,num_blocks=1).to(device)
        elif component == 'phasedLSTM':
            self.layer = PhasedLSTM(input_size, hidden_size, num_layers,batch_first=True).to(device)
        else:
            self.layer = nn.RNN(input_size, hidden_size, num_layers, batch_first=True).to(device)
            
        self.linear = nn.Linear(hidden_size, out_size).to(device)

    def forward(self, input_seq, h=None, c=None, m=None):
        # batch_size, seq_len, input_size = input_seq.shape
        # input_seq = input_seq.squeeze(1)
        batch_size,seq_len, feature_size = input_seq.shape
        times = torch.arange(feature_size).to(self.device)


        # 确保mask维度正确
        a_x = m[:, 0].unsqueeze(-1) # (batch_size, seq_len)
        a_h = m[:, 1].unsqueeze(-1)  # (batch_size, seq_len)
        a_y = m[:, 2].unsqueeze(-1)  # (batch_size, seq_len)


        # 应用输入mask
        x = input_seq * a_x.unsqueeze(-1)  # (batch_size, seq_len, input_size)
        component_list = ['LSTM', 'PeepholeLSTM', 'indRNN', 'xLSTM','phasedLSTM']
        if self.component in component_list:
            # 初始化隐藏状态
            if h is None:
                h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
            if c is None:
                c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

            # 根据mask更新状态
            mask = a_h.unsqueeze(0).expand(self.num_layers, -1, -1)  # (num_layers, batch_size, seq_len)
            if self.component == 'phasedLSTM':
                output, (h_new, c_new) = self.layer(x, (h, c), times)
            else:
                output, (h_new, c_new) = self.layer(x, (h, c)) #b,t,h
            
            # 选择性更新
            h = torch.where(mask == 1, h_new, h)
            c = torch.where(mask == 1, c_new, c)

        else:  # GRU or RNN
            if h is None:
                h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

            mask = a_h.unsqueeze(0).expand(self.num_layers, -1, -1)
            output, h_new = self.layer(x, h)
            
            # 选择性更新
            h = torch.where(mask == 1, h_new, h)

        # 应用线性层和输出mask
        y = self.linear(output)  # (batch_size, seq_len, out_size)
        y = y * a_y.unsqueeze(-1)  # 应用输出mask

        if self.component in self.component_list:
            return h, c, y
        else:
            return h, None, y

