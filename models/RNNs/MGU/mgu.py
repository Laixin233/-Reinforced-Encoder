import torch
import torch.nn as nn
import torch.nn.functional as F

class MGUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MGUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 门参数分解为输入相关的权重和隐藏状态相关的权重
        self.weight_ih = nn.Parameter(torch.Tensor(2 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(2 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(2 * hidden_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """根据论文建议初始化忘记门偏置"""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)
        # 忘记门偏置初始化为1.0（对应第二个门参数）
        nn.init.constant_(self.bias_ih[self.hidden_size:], 1.0)
        nn.init.constant_(self.bias_hh[self.hidden_size:], 1.0)
        
    def forward(self, input, hx):
        # input: (batch_size, input_size)
        # hx: (batch_size, hidden_size)
        
        # 合并线性变换
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih + \
                torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        
        # 分割门参数
        f_gate = torch.sigmoid(gates[:, :self.hidden_size])  # 忘记门
        h_tilde = torch.tanh(gates[:, self.hidden_size:])    # 候选隐藏状态
        
        # 计算新的隐藏状态
        h_new = (1 - f_gate) * hx + f_gate * h_tilde
        return h_new

class MGU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(MGU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.cells.append(MGUCell(in_size, hidden_size))
            
    def forward(self, x, hx=None):
        if self.batch_first:
            x = x.transpose(0, 1)  # (batch, seq, features) -> (seq, batch, features)
            
        seq_len, batch_size, _ = x.size()
        
        if hx is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        else:
            h0 = hx
            
        h = h0
        layer_outputs = []
        
        for t in range(seq_len):
            layer_input = x[t]
            new_h = []
            for layer in range(self.num_layers):
                cell = self.cells[layer]
                h_layer = h[layer]
                h_next = cell(layer_input, h_layer)
                new_h.append(h_next)
                layer_input = h_next  # 下一层的输入是当前层的输出
            h = torch.stack(new_h, dim=0)
            layer_outputs.append(h[-1])  # 只记录最后一层的输出
            
        outputs = torch.stack(layer_outputs, dim=0)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)  # (seq, batch, features) -> (batch, seq, features)
            
        return outputs, h

# 测试示例
if __name__ == "__main__":
    # 超参数
    input_size = 10
    hidden_size = 32
    num_layers = 2
    batch_size = 4
    seq_len = 8
    batch_first = True
    
    # 创建模型
    model = MGU(input_size, hidden_size, num_layers, batch_first)
    
    # 生成测试数据
    if batch_first:
        x = torch.randn(batch_size, seq_len, input_size)
    else:
        x = torch.randn(seq_len, batch_size, input_size)
    
    # 前向传播
    outputs, hidden = model(x)
    print(f"Output shape: {outputs.shape}")  # 预期输出: (batch_size, seq_len, hidden_size)
    print(f"Hidden state shape: {hidden.shape}")  # 预期输出: (num_layers, batch_size, hidden_size)