import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli,Categorical
from torch.nn import TransformerEncoder, TransformerEncoderLayer

#Actor-critic中的actor网络
class AgentMLP(nn.Module):
    ## 离散空间采用了 softmax policy 来参数化策略
    def __init__(self, obs_n, device, hidden_size=128, dropout_rate=0.6,jump_step=5):
        super(AgentMLP, self).__init__()
        self.affline1 = nn.Linear(obs_n, hidden_size).to(device)
        self.dropout1 = nn.Dropout(p=dropout_rate).to(device)
        self.affline2 = nn.Linear(hidden_size, hidden_size).to(device)
        self.dropout2 = nn.Dropout(p=dropout_rate).to(device)
        # self.affline3 = nn.Linear(hidden_size, 3).to(device)  # 输出层调整为输出3个动作的概率
        
        # a 和 b 的输出层（二值分类）
        self.a_output = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        ).to(device)
        self.b_output = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        ).to(device)
        # c 的输出层（多分类）
        self.c_output = nn.Sequential(
            nn.Linear(hidden_size, jump_step),
            nn.Softmax(dim=1)
        ).to(device)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affline1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.affline2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        a = self.a_output(x)
        b = self.b_output(x)
        c = self.c_output(x)
        # probs = torch.cat((a, b, c), dim=1)
        return [a,b,c]

class AgentTransformer(nn.Module):
    def __init__(self, obs_n, hidden_size=128, dropout_rate=0.6,  nhead=4, num_layers=2, jump_step=5,device=None):
        super(AgentTransformer, self).__init__()
        self.device = device
        
        # 初始特征提取层（保持输入维度兼容）
        self.initial_fc = nn.Sequential(
            nn.Linear(obs_n, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ).to(device)
        
        # Transformer编码器层
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True  # 输入为 (batch, seq_len, features)
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers).to(device)
        
        # 输出头（保持与原结构一致）
        self.a_output = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        ).to(device)
        self.b_output = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        ).to(device)
        self.c_output = nn.Sequential(
            nn.Linear(hidden_size, jump_step),
            nn.Softmax(dim=1)
        ).to(device)
        
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, state):
        # 输入x形状: (batch_size, obs_n)
        batch_size = state.size(0)
        
        # Step 1: 初始特征提取
        x = self.initial_fc(state)  # (batch_size, hidden_size)
        
        # Step 2: 添加虚拟序列维度以适应Transformer
        # Transformer期望输入为 (batch, seq_len, features)，此处seq_len=1
        # x = x.unsqueeze(1)     # (batch_size, 1, hidden_size)

        x = self.transformer_encoder(x)  # (batch_size, 1, hidden_size)
        # x = x.squeeze(1)       # (batch_size, hidden_size)
        
        # Step 4: 输出头（保持与原结构一致）
        a = self.a_output(x)
        b = self.b_output(x)
        c = self.c_output(x)
        
        # 训练agent时要考虑trans的维度
        if len(c.shape) == 3:
            action = torch.cat((a, b, c), dim=2)
        else:
            action = torch.cat((a, b, c), dim=1)
        return action #[a, b, c] (batch_size, (1+1+jump_step))
    


class PolicyAgentNet(nn.Module):
    def __init__(self, obs_n, device, hidden_size=None, dropout_rate=0.6, learning_rate=1e-2, step_gamma=0.99, gamma=0.99, sample_rate=0.1,jump_step=4):
        super(PolicyAgentNet, self).__init__()
        self.obs_n = obs_n
        self.device = device
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        self.policy = AgentTransformer(obs_n, device, hidden_size, dropout_rate,jump_step=jump_step)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.eps = np.finfo(np.float32).eps.item()  # 非负的最小值，使得归一化时分母不为0
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=step_gamma)
        self.gamma = gamma
    
    def select_action(self, state,t,generate_action=False):
        if t == 0:
            self.policy.saved_log_probs.clear()
        probs = self.policy(state)
        a_probs,b_probs,c_probs = probs
        
        m_a = Bernoulli(a_probs)  
        m_b = Bernoulli(b_probs)  # 为每个动作生成伯努利分布
        m_c = Categorical(c_probs) #分布

        # if generate_action:
        #     action = torch.ones_like(probs)
        # else:
        a = m_a.sample()  # 从分布中采样
        b = m_b.sample()
        c = m_c.sample().unsqueeze(1)
        action = torch.concatenate([a,b,c],dim=1)
        log_prob = m_a.log_prob(a) + m_b.log_prob(b) + m_c.log_prob(c)
        self.policy.saved_log_probs.append(log_prob)  # 取对数似然 logπ(s,a)
        return action
    
    def predict(self, state):
        probs = self.policy(state)
        action = torch.argmax(probs, dim=1)
        return action
    
    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, torch.tensor(R, device=self.device))
        returns = torch.stack(returns, dim=1) #b,t,1->b,1
        saved_log_probs = torch.stack(self.policy.saved_log_probs, dim=1) 
        cross_entropy_loss = torch.sum(saved_log_probs, dim=2) # 交叉熵 (T, 3)
        record_returns = torch.mean(returns)
        returns = (returns - torch.mean(returns)) / (torch.std(returns) + self.eps)
        returns = returns.squeeze()
        self.optimizer.zero_grad()
        policy_loss = -torch.sum(torch.mul(cross_entropy_loss, returns))
        policy_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
        
        return record_returns.item()
    
    def init_finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, torch.tensor(R, device=self.device))
        returns = torch.stack(returns, dim=1)
        saved_log_probs = torch.stack(self.policy.saved_log_probs, dim=1) 
        cross_entropy_loss = torch.sum(saved_log_probs, dim=2) # 交叉熵 (T, 3)
        record_returns = torch.mean(returns)
        returns = (returns - torch.mean(returns)) / (torch.std(returns) + self.eps)
        returns = returns.squeeze()
        self.optimizer.zero_grad()
        # policy_loss = -torch.sum(torch.mul(1, returns))
        policy_loss = -torch.sum(torch.mul(cross_entropy_loss, returns))
        policy_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
        
        return record_returns.item()
    
    def reset_learning(self):
        self.optimizer.param_groups[0]['lr'] = self.learning_rate





#rnn策略网络
class AgentRNN(nn.Module):
    def __init__(self, obs_n, hidden_size=128, num_layers=2, device=None, dropout_rate=0.6):
        super(AgentRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.layer = nn.GRU(obs_n, hidden_size, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size*num_layers, 3).to(device)
        self.saved_log_probs = []
        self.rewards = []
    
    def forward(self, x,h=None):
        batch_size = x.shape[0]
        if h is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        out, h_new = self.layer(x,h)
        x_out = self.fc(out.reshape(batch_size,-1))
        probs = torch.sigmoid(x_out)  # 使用sigmoid输出0或1的概率  
        return probs, h_new
    
#value网络
class QvalueMLPNet(nn.Module):
    ## 离散空间采用了 softmax policy 来参数化策略
    def __init__(self, obs_n, num_actions, hidden_size=128, dropout_rate=0.6):
        super(QvalueMLPNet, self).__init__()
        self.affline1 = nn.Linear(obs_n, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.affline2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.affline3 = nn.Linear(hidden_size, num_actions)  # 输出层调整为输出num_actions个动作的Q值
        
    def forward(self, x):
        x = self.affline1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.affline2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        qvalue = self.affline3(x)
        return qvalue


class QValueTranformerNet(nn.Module):
    def __init__(self, obs_n, hidden_size=128, dropout_rate=0.6,  nhead=4, num_layers=2, device=None):
        super(QValueTranformerNet, self).__init__()
        self.device = device
        
        
        # 初始特征提取层（保持输入维度兼容）
        self.initial_fc = nn.Sequential(
            nn.Linear(obs_n, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ).to(device)
        
        # Transformer编码器层
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * nhead,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True  # 输入为 (batch, seq_len, features)
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers).to(device)
        
        # 输出头（保持与原结构一致）
        self.output = nn.Sequential(
            nn.Linear(hidden_size,1),
            nn.Sigmoid()
        ).to(device)
        
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        # 输入x形状: (num_trans,batch_size, obs_n)
        
        x = self.initial_fc(x)  # (batch_size, hidden_size)
        x = self.transformer_encoder(x)  # (batch_size, 1, hidden_size)
        qvalue = self.output(x)

        return qvalue


#rnn评估网络critic net
class QValuernnNet(nn.Module):
    def __init__(self, obs_n, num_actions,hidden_size=128, num_layers=2, dropout_rate=0.6, device=None):
        super(QValuernnNet, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.layer = nn.GRU(obs_n, hidden_size, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size*num_layers, num_actions).to(device)
        self.saved_log_probs = []
        self.rewards = []
    
    def forward(self, x,h=None):
        batch_size = x.shape[0]
        if h is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        out, h_new = self.layer(x,h)
        x_out = self.fc(out.reshape(batch_size,-1))
        probs = torch.sigmoid(x_out)  # 使用sigmoid输出0或1的概率  
        return probs, h_new

    

class PolicyAgentRnn(nn.Module):
    def __init__(self, obs_n,  hidden_size, num_layers, device, dropout_rate=0.6, learning_rate=1e-2, step_gamma=0.99, gamma=0.99, sample_rate=0.1):
        super(PolicyAgentRnn, self).__init__()
        self.obs_n = obs_n
        self.device = device
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        self.policy = PolicyRNN(obs_n, hidden_size, num_layers=num_layers, device=device, dropout_rate=dropout_rate)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.eps = np.finfo(np.float32).eps.item()  # 非负的最小值，使得归一化时分母不为0
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=step_gamma)
        self.gamma = gamma
    
    def select_action(self, state,t,h=None,generate_action=False):
        if t == 0:
            self.policy.saved_log_probs.clear()
        probs,h = self.policy(state,h)
        m = Bernoulli(probs)  # 为每个动作生成伯努利分布
        if generate_action:
            action = torch.ones_like(probs)
        else:
            action = m.sample()  # 从分布中采样动作
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action,h
    
    def predict(self, state,h):
        probs = self.policy(state,h)
        action = torch.argmax(probs, dim=1)
        return action
    
    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, torch.tensor(R, device=self.device))
        returns = torch.stack(returns, dim=1) #b,t,1->b,1
        saved_log_probs = torch.stack(self.policy.saved_log_probs, dim=1) 
        cross_entropy_loss = torch.sum(saved_log_probs, dim=2) # 交叉熵 (T, 3)
        record_returns = torch.mean(returns)
        returns = (returns - torch.mean(returns)) / (torch.std(returns) + self.eps)
        returns = returns.squeeze()
        self.optimizer.zero_grad()
        policy_loss = -torch.sum(torch.mul(cross_entropy_loss, returns))
        policy_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
        
        return record_returns.item()
    
    def init_finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, torch.tensor(R, device=self.device))
        returns = torch.stack(returns, dim=1)
        saved_log_probs = torch.stack(self.policy.saved_log_probs, dim=1) 
        cross_entropy_loss = torch.sum(saved_log_probs, dim=2) # 交叉熵 (T, 3)
        record_returns = torch.mean(returns)
        returns = (returns - torch.mean(returns)) / (torch.std(returns) + self.eps)
        returns = returns.squeeze()
        self.optimizer.zero_grad()
        # policy_loss = -torch.sum(torch.mul(1, returns))
        policy_loss = -torch.sum(torch.mul(cross_entropy_loss, returns))
        policy_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
        
        return record_returns.item()
    
    def reset_learning(self):
        self.optimizer.param_groups[0]['lr'] = self.learning_rate
    


