import torch
import torch.nn as nn

class PhasedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, alpha=0.001, rho=0.05, train_alpha=False, train_rho=False):
        super(PhasedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.alpha = torch.tensor([alpha], requires_grad=train_alpha)
        self.rho = torch.tensor([rho], requires_grad=train_rho)
        self.tau = nn.Parameter(torch.randn(hidden_size) * 0.5 + 5.0)  # 定义周期参数
        self.s = nn.Parameter(torch.rand(hidden_size) * self.tau)  # 定义相位偏移参数

        # 定义LSTM权重
        self.Wxh = nn.Linear(input_size, 4 * hidden_size)
        self.Whh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input,time,states):

        hx, cx = states
        gates = self.Wxh(input) + self.Whh(hx)
        i, f, g, o = gates.chunk(4, 1)
        
        # 时间门控计算
        with torch.no_grad():
            phi = torch.fmod((time - self.s), self.tau) / self.tau
            k = torch.zeros_like(phi).to(input.device)
            self.rho = self.rho.to(input.device)
            self.alpha = self.alpha.to(input.device)
            # 条件判断部分
            t0 = phi < self.rho / 2.0
            k[t0] = 2 * phi[t0] / self.rho
            t1 = (phi >= self.rho / 2.0) & (phi < self.rho)
            k[t1] = 2 - 2 * phi[t1] / self.rho
            t2 = phi >= self.rho
            k = torch.where(t2, self.alpha * phi, k)

        # 更新细胞状态和隐藏状态
        ft = self.sigmoid(f + 1.0 - k)
        it = self.sigmoid(i) * k
        gt = self.tanh(g) * k
        ot = self.sigmoid(o) * k
        
        cy = (ft * cx) + (it * gt)
        hy = ot * self.tanh(cy)
        return hy, cy

class PhasedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, alpha=0.001, rho=0.05, train_alpha=False, train_rho=False, batch_first=False):
        super(PhasedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(PhasedLSTMCell(input_size if _ == 0 else hidden_size, hidden_size, alpha, rho, train_alpha, train_rho))
    
    def forward(self, inputs, hc=None,times=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)  # 转置为 seq_len x batch_size x input_size

        
        seq_len, batch_size, _ = inputs.size()
        if hc is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(inputs.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(inputs.device)
        else:
            h0, c0 = hc
        h, c = h0, c0
        outputs = []
        for i in range(seq_len):
            new_h = []
            new_c = []
            for layer_idx in range(self.num_layers):
                if layer_idx == 0:
                    current_input = inputs[i]
                else:
                    current_input = prev_h
                time = times[i]
                prev_h, prev_c = h[layer_idx], c[layer_idx]
                layer_out = self.layers[layer_idx](current_input, time, (prev_h, prev_c))
                new_h.append(layer_out[0])
                new_c.append(layer_out[1])
                prev_h = layer_out[0]
            h, c = torch.stack(new_h), torch.stack(new_c)
            outputs.append(h[-1])  # 取最后一层的输出
        
        outputs = torch.stack(outputs)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)  # 转置回 batch_first 的顺序
        
        return outputs, (h, c)