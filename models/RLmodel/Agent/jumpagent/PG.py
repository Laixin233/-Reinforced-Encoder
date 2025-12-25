# 策略梯度算法
# https://blog.csdn.net/qq_39160779/article/details/107295128

import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class PolicyGAgent():
    def __init__(self, obs_n, act_n, device, hidden_size = 128,dropout_rate = 0.6,learning_rate=1e-2, step_gamma = 0.99,gamma=0.99,sample_rate = 0.1):
        self.obs_n = obs_n
        self.act_n = act_n
        self.gamma = gamma
        self.device = device
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        self.policy = Policy(obs_n, act_n,device,hidden_size,dropout_rate)
        self.optimizer = optim.Adam(self.policy.parameters(),lr=learning_rate)
        self.eps = np.finfo(np.float32).eps.item()  # 非负的最小值，使得归一化时分母不为0
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=step_gamma)
    
    def select_action(self,state):
        ## 选择动作，这个动作不是根据Q值来选择，而是使用softmax生成的概率来选
        #  不需要epsilon-greedy，因为概率本身就具有随机性
        # state = torch.from_numpy(state).float().unsqueeze(0)
        #print(state.shape)   torch.size([1,4])
        probs = self.policy(state)
        m = Categorical(probs)      # 生成分布
        action = m.sample()           # 从分布中采样
        
        
        # for i in range(action.size(0)):
        #     if random.random() < self.sample_rate:
        #         action[i] = random.choice([q for q in range(self.act_n)])
                
                
        #print(m.log_prob(action))   # m.log_prob(action)相当于probs.log()[0][action.item()].unsqueeze(0)
        self.policy.saved_log_probs.append(m.log_prob(action))    # 取对数似然 logπ(s,a)
        # return action.item()         # 返回一个元素值
    
        # return_action = torch.zeros((action.size(0),self.act_n))
        # for i in range(action.size(0)):
        #     return_action[i,action[i]] = 1
        if action.is_cuda:
            return torch.cuda.LongTensor(action).unsqueeze(1)
        else:
            return torch.LongTensor(action).unsqueeze(1)
    
    def predict(self,state):
        #输入：state为ht（batch_size,hidden_size）
        #输出：动作action
        ## 选择动作，这个动作不是根据Q值来选择，而是使用softmax生成的概率来选
        #  不需要epsilon-greedy，因为概率本身就具有随机性
        # state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state) #（batch_size,act_n）
        action =torch.max(probs,1)[1]
        # return action.item()         # 返回一个元素值
    
        # return_action = np.zeros((action.size(0),self.act_n))
        # for i in range(action.size(0)):
        #     return_action[i][action[i]] = 1
        if action.is_cuda:
            return torch.cuda.LongTensor(action).unsqueeze(1)
        else:
            return torch.LongTensor(action).unsqueeze(1)
    
    
    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0,R)        # 将R插入到指定的位置0处
        returns = torch.stack(returns,dim = 1) #batch_size,seq_len
        saved_log_probs = torch.stack(self.policy.saved_log_probs,dim = 1)
        record_retruns = torch.mean(returns)    # 记录当前的rewars
        returns = (returns - torch.mean(returns)) / (torch.std(returns) + self.eps)     # 归一化
        self.optimizer.zero_grad()
        policy_loss = - torch.sum(torch.mul(saved_log_probs,returns))   # 损失函数为交叉熵 
        policy_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        del self.policy.rewards[:]          # 清空episode 数据
        del self.policy.saved_log_probs[:]
        
        return record_retruns.item()
    
    def reset_learning(self):
        self.optimizer.param_groups[0]['lr'] = self.learning_rate

class Policy(nn.Module):
    ##  离散空间采用了 softmax policy 来参数化策略
    def __init__(self,obs_n, act_n,device,hidden_size = 128,dropout_rate = 0.6):
        super(Policy,self).__init__()
        self.affline1 = nn.Linear(obs_n,hidden_size).to(device)
        self.dropout = nn.Dropout(p=dropout_rate).to(device)
        self.affline2 = nn.Linear(hidden_size,act_n).to(device)

        self.saved_log_probs = []
        self.rewards = []
    def forward(self, x):
        x = self.affline1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affline2(x)
        return F.softmax(action_scores,dim=1)
