import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class PeepholeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PeepholeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入门、遗忘门、输出门的权重和偏置
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

        # Peephole 连接的权重
        self.weight_ic = nn.Parameter(torch.randn(hidden_size))
        self.weight_fc = nn.Parameter(torch.randn(hidden_size))
        self.weight_oc = nn.Parameter(torch.randn(hidden_size))

    def forward(self, input, hx):
        h, c = hx
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(h, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # Peephole 连接
        ingate = torch.sigmoid(ingate + c * self.weight_ic)
        forgetgate = torch.sigmoid(forgetgate + c * self.weight_fc)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate + c * self.weight_oc)

        c_new = (forgetgate * c) + (ingate * cellgate)
        h_new = outgate * torch.tanh(c_new)

        return h_new, c_new

class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(PeepholeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = nn.ModuleList([
            PeepholeLSTMCell(input_size, hidden_size) if i == 0 else PeepholeLSTMCell(hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        # self.output_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, x, hx=None):
        if self.batch_first:
            x = x.transpose(0, 1)  # (batch, seq, feature) -> (seq, batch, feature)

        seq_len, batch_size, _ = x.size()
        if hx is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        else:
            h0, c0 = hx

        h, c = h0, c0
        outputs = []

        for t in range(seq_len):
            h_layer = []
            c_layer = []
            for i, cell in enumerate(self.cells):
                if i == 0:
                    h_i, c_i = cell(x[t], (h[i].detach(), c[i].detach()))
                else:
                    h_i, c_i = cell(h[i-1].detach(), (h[i].detach(), c[i].detach()))
                h_layer.append(h_i)
                c_layer.append(c_i)
            h = torch.stack(h_layer, 0)
            c = torch.stack(c_layer, 0)
            outputs.append(h[-1])

        outputs = torch.stack(outputs, 0)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)  # (seq, batch, feature) -> (batch, seq, feature)
        # outputs = self.output_layer(outputs)

        return outputs, (h, c)

def generate_data(num_samples, seq_length, input_size):
    x = np.random.randn(num_samples, seq_length, input_size)
    y = np.random.randn(num_samples, seq_length, 1)  # 假设输出是一个标量
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

#仅在本文件测试
def __main__():

    # 超参数
    input_size = 10
    hidden_size = 50
    num_layers = 2
    batch_first = True
    seq_length = 20
    batch_size = 32
    num_samples = 1000

    # 生成数据
    x, y = generate_data(num_samples, seq_length, input_size)

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 创建 DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = PeepholeLSTM(input_size, hidden_size, num_layers, batch_first)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # 前向传播
            outputs, _ = model(batch_x)

            # 计算损失
            loss = criterion(outputs, batch_y)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            predictions, _ = model(batch_x)
            test_loss = criterion(predictions, batch_y)
            test_losses.append(test_loss.item())

    test_loss = np.mean(test_losses)
    print(f'Test Loss: {test_loss:.4f}')