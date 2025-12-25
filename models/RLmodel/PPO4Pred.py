import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.optim.lr_scheduler import StepLR
# from torch.utils.tensorboard import SummaryWriter
from models.RNNs.maskRNN import MaskEncoder
from models.RLmodel.Agent.jumpagent.agentEnv import Environment
from models.RLmodel.Agent.jumpagent.PolicyGradient import AgentTransformer,QValueTranformerNet,AgentRNN,QValuernnNet,AgentMLP,QvalueMLPNet    
# from models.RLmodel.Agent.nonpatchagent.agentEnv import Environment
from task.TaskLoader import Opt
from tqdm import tqdm
import os
import copy
import gc
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from torch.utils.data.sampler import BatchSampler,SubsetRandomSampler
from models.RLmodel.Agent.sampletrans.DynamicSample import DynamicTransitionSampler
from collections import namedtuple
from torch.distributions import Bernoulli,Categorical

from torch.utils.data import DataLoader, TensorDataset

import numpy as np




class ppoAgent(nn.Module):
    def __init__(self, opt=None, logger=None):
        super(ppoAgent, self).__init__()
        self.opt = opt
        self.logger = logger

        self.device = opt.device
        self.epochs = opt.epochs
        self.H = opt.H
        self.num_actions = opt.num_actions
        self.agent_model = opt.agent_model
        self.preTrain_epochs = opt.preTrain_epochs
        self.batch_size = opt.batch_size
        self.clip_param = opt.clip_param
        self.max_grad_norm = opt.max_grad_norm
        self.buffer_capacity = opt.buffer_capacity
        self.agent_iter = opt.agent_iter
        num_states = self.opt.encoder_hidden_size * self.opt.encoder_num_layer + self.opt.num_variate
        #agent
        if self.agent_model == 'Transformer':
            
            self.actor_net =   AgentTransformer(num_states,hidden_size=opt.agent_hidden_size,dropout_rate=opt.agent_dropout_rate,nhead=opt.agent_nhead, num_layers=opt.agent_num_layer,jump_step=opt.jump_step,device=self.device).to(self.device)
            self.critic_net = QValueTranformerNet(num_states,hidden_size=opt.agent_hidden_size,dropout_rate=opt.agent_dropout_rate,nhead=opt.agent_nhead, num_layers=opt.agent_num_layer,device=self.device).to(self.device)
        elif self.agent_model == 'MLP':
            self.actor_net = AgentMLP(num_states, opt.num_actions, hidden_size=opt.agent_hidden_size, dropout=opt.agent_dropout_rate).to(self.device)
            self.critic_net = QvalueMLPNet(num_states, opt.num_actions, hidden_size=opt.agent_hidden_size, dropout=opt.agent_dropout_rate).to(self.device)        
        else:
            num_states = self.opt.encoder_hidden_size
            self.actor_net = AgentRNN(num_states,self.num_actions,hidden_size=opt.encoder_hidden_size,num_layers=opt.encoder_num_layer,dropout_rate=opt.dropout_rate,device=self.device).to(self.device)
            self.critic_net = QValuernnNet(num_states,self.num_actions,hidden_size=opt.encoder_hidden_size,num_layers=opt.encoder_num_layer,dropout_rate=opt.dropout_rate,device=self.device).to(self.device)

        self.dts_sampler = DynamicTransitionSampler(opt, self.device)
        self.predictions_buffer = []
        self.targets_buffer = []

        self.buffer=[]
        self.counter = 0
        self.training_step = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=opt.agent_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=opt.agent_lr)
        self.loss_func = nn.MSELoss()
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=opt.step_gamma)
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=opt.step_gamma)
        
    
    
    def choose_action(self, state):
        with torch.no_grad():
            probs = self.actor_net(state)
        a_probs,b_probs,c_probs = probs[:,0],probs[:,1],probs[:,2:]
        
        m_a = Bernoulli(a_probs)  
        m_b = Bernoulli(b_probs)  # 为每个动作生成伯努利分布
        m_c = Categorical(c_probs) #分布

        a = m_a.sample()  # 从分布中采样
        b = m_b.sample()
        c = m_c.sample()

        a_probs = m_a.log_prob(a)
        b_probs = m_b.log_prob(b)
        c_probs = m_c.log_prob(c) #取对数似然,动作的概率
        action = torch.stack([a,b,c],dim=1)
        action_log_prob = torch.stack([a_probs,b_probs,c_probs],dim=1) 

        return action.detach(),action_log_prob.detach()   


    def get_action_probs(self, state):
        # 有梯度计算，这一步是放进池子里，所以三维
        probs = self.actor_net(state)
        a_probs,b_probs,c_probs = probs[:,:,0],probs[:,:,1],probs[:,:,2:]
        
        m_a = Bernoulli(a_probs)  
        m_b = Bernoulli(b_probs)  # 为每个动作生成伯努利分布
        m_c = Categorical(c_probs) #分布

        a = m_a.sample()  # 从分布中采样
        b = m_b.sample()
        c = m_c.sample()
        action = torch.stack([a,b,c],dim=1)
        a_probs = m_a.log_prob(a)
        b_probs = m_b.log_prob(b)
        c_probs = m_c.log_prob(c) #取对数似然,动作的概率
        action_log_prob = torch.stack([a_probs,b_probs,c_probs],dim=2) 

        return action_log_prob

    def get_value(self, state):
        with torch.no_grad():
            value = self.critic_net(state)
        return value
    

    
    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.counter % self.buffer_capacity
        self.buffer[index] = transition
        self.counter += 1
    
    def get_log_prob(self, state, action):
        action_prob = self.actor_net(state)
        dist = Categorical(action_prob)
        action_log_prob = dist.log_prob(action)
        return action_log_prob
    
    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter+=1
        return self.counter % (self.batch_size*10) == 0


    def update(self):
        """
        PPO 更新 - 使用动态转换采样
        """
        if len(self.predictions_buffer) > 0 and len(self.targets_buffer) > 0:
            predictions = torch.stack(self.predictions_buffer)  # (M, H)
            targets = torch.stack(self.targets_buffer)  # (M, H)
        else:
            predictions = None
            targets = None
        
        # 提取完整的缓冲区数据（用于计算 TD 误差和优先级）
        state = torch.stack([t.state for t in self.buffer])
        action = torch.stack([t.action for t in self.buffer])
        reward = torch.stack([t.reward for t in self.buffer])
        next_state = torch.stack([t.next_state for t in self.buffer])
        old_action_log_prob = torch.stack([t.action_log_prob for t in self.buffer])
        
        # 标准化奖励
        reward = (reward - reward.mean()) / (reward.std() + 1e-10)
        
        # 计算目标价值和优势（在完整数据上）
        with torch.no_grad():
            next_value = self.critic_net(next_state)
            target_value = reward + self.opt.gamma * next_value
        
        advantage = target_value - self.get_value(state).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
        
        # 这里是内部循环，训练 agent_iter 次
        for g in range(self.agent_iter): 
            
            # 每次循环重新采样
            sampled_indices, sampled_transitions = self.dts_sampler.sample_transitions(
                self.buffer,
                num_samples=self.opt.batch_size,  #   采样 batch_size 个样本
                critic_net=self.critic_net,
                current_epoch=g,  #   传入当前内部循环的 g
                predictions=predictions,
                targets=targets
            )
            
            # 从采样的索引中提取数据
            sampled_state = state[sampled_indices]
            sampled_action = action[sampled_indices]
            sampled_reward = reward[sampled_indices]
            sampled_next_state = next_state[sampled_indices]
            sampled_old_log_prob = old_action_log_prob[sampled_indices]
            sampled_target_value = target_value[sampled_indices]
            sampled_advantage = advantage[sampled_indices]


            #   policy loss 
            action_log_prob = self.get_action_probs(sampled_state.unsqueeze(0)).squeeze(0)
            ratio = torch.exp(action_log_prob - sampled_old_log_prob)
            surr1 = ratio * sampled_advantage
            surr2 = torch.clamp(
                ratio, 1 - self.clip_param, 1 + self.clip_param
            ) * sampled_advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Actor 更新
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_net.parameters(), self.max_grad_norm
            )
            self.actor_optimizer.step()
            
            #  计算 value loss 
            critic_loss = self.loss_func(
                self.critic_net(sampled_state), sampled_target_value
            )
            
            # Critic 更新
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic_net.parameters(), self.max_grad_norm
            )
            self.critic_optimizer.step()
            
            self.training_step += 1
        
        #   Update weight parameters
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # 清空缓冲区
        del self.buffer[:]
        del self.predictions_buffer[:]
        del self.targets_buffer[:]


class rnnPPO(nn.Module):
    def __init__(self, opt=None, logger=None):
        super(rnnPPO, self).__init__()
        self.opt = opt
        self.logger = logger

        self.device = opt.device
        self.epochs = opt.epochs
        self.H = opt.H
        self.T = opt.T

        #self.L = #self.T - opt.patch_size +1
        self.num_actions = opt.num_actions
        self.preTrain_epochs = opt.preTrain_epochs
        self.batch_size = opt.batch_size
        self.clip_param = opt.clip_param
        self.max_grad_norm = opt.max_grad_norm
        self.buffer_capacity = opt.buffer_capacity
        self.ppo_epochs = opt.ppo_epochs

        self.ppoagent = ppoAgent(opt,logger)
        self.env = Environment(opt,logger)

        self.fit_info = Opt()
        self.fit_info.min_vrmse = float('inf')
        self.fit_info.loss_list = []
        self.fit_info.vloss_list = []
        self.fit_info.t_mse_list = []
        self.fit_info.v_mse_list = []
        self.fit_info.encoder_loss_list = []
        self.fit_info.encoder_vloss_list = []

        self.best_environment = None
        self.best_policy = None
        self.best_state = None
        self.Transition = namedtuple('Transition',['state', 'action', 'reward', 'action_log_prob', 'next_state'])
        self.TrainRecord = namedtuple('TrainRecord',['episode', 'reward'])
        self.loss_func = nn.MSELoss()
        self.best_mdplist = None

    def xfit(self, train_loader, val_loader):
        self.logger.info("Start training...")
        self.logger.info('agent_T_agent_rate: {}\t output_agent:  {}'.format(self.opt.error_rate, self.opt.output_agent))
        self.logger.info('Number of training examples: {}'.format(len(train_loader.dataset)))
        vli_list =self.env.initial_environment(train_loader, val_loader,self.preTrain_epochs)
        self.fit_info.min_vrmse = vli_list[0]
        self.best_environment = copy.deepcopy(self.env.state_dict())
        self.best_critic = copy.deepcopy(self.ppoagent.critic_net.state_dict())
        self.best_actor = copy.deepcopy(self.ppoagent.actor_net.state_dict())
        N_badnet = 0
        for epoch in range(self.epochs):
            self.logger.info("training epoch: {}".format(epoch))
            if epoch >0:
                self.train_environment(train_loader,self.opt.env_iter)
            MDP_lists = self.train_agent(train_loader, val_loader)
            MDP_lists, _, trmse, vrmse = self.evaluate_encoder(train_loader,val_loader)#MDP_lists: (num_data, seq_len, num_actions)

            if vrmse < self.fit_info.min_vrmse :
                self.fit_info.min_vrmse = vrmse
                self.fit_info.trmse = trmse
                self.fit_info.vrmse = vrmse
                self.best_epoch = epoch
                self.best_critic = copy.deepcopy(self.ppoagent.critic_net.state_dict())
                self.best_actor = copy.deepcopy(self.ppoagent.actor_net.state_dict())
                self.best_environment = copy.deepcopy(self.env.state_dict())
                self.best_mdplist = MDP_lists
            else:
                N_badnet = N_badnet+1                    
                if N_badnet > self.opt.patience:
                    self.env.load_state_dict(self.best_environment)
                    self.ppoagent.critic_net.load_state_dict(self.best_critic)
                    self.ppoagent.actor_net.load_state_dict(self.best_actor)
                    N_badnet = 0
                    batch_size, seq_len, feature_size = concentrateLoader(train_loader, self.device) 
                    # self.best_mdplist = torch.ones(batch_size, seq_len, feature_size).to(self.device)
                    
            self.xfit_logger(epoch)
            torch.cuda.empty_cache() if next(self.env.parameters()).is_cuda else gc.collect()
            torch.cuda.empty_cache() if next(self.ppoagent.actor_net.parameters()).is_cuda else gc.collect()
            torch.cuda.empty_cache() if next(self.ppoagent.critic_net.parameters()).is_cuda else gc.collect()
        

        return self.fit_info  

    def train_environment(self, train_loader, iter):
        """训练环境（Encoder）- 使用 agent 选择的动作"""
        for epoch in range(iter):
            self.logger.info(f"environment training epoch: {epoch}")
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                data_x, data_y = batch_x.detach().clone(), batch_y.detach().clone()
                batch_size = data_x.shape[0]
                _, T, _ = batch_x.size()
                
                # 初始化
                h_0 = torch.zeros(
                    self.opt.encoder_num_layer,
                    batch_size,
                    self.opt.encoder_hidden_size
                ).to(self.device)
                c_0 = torch.zeros(
                    self.opt.encoder_num_layer,
                    batch_size,
                    self.opt.encoder_hidden_size
                ).to(self.device)
                
                h_t = h_0
                c_t = c_0
                H_history = []
                
                pred_list = []
                a_y_list = []  # 记录每个时间步的 a_y (q_t)
                
                # 前向传播
                for t in range(T):
                    data_x_in = data_x[:, t, :].unsqueeze(1)
                    
                    # 构造状态并选择动作
                    h_flat = h_t.permute(1, 0, 2).reshape(batch_size, -1)
                    x_flat = data_x[:, t, :]
                    state = torch.cat([h_flat, x_flat], dim=1)
                    
                    action, _ = self.ppoagent.choose_action(state)
                    
                    #   提取动作
                    a_x = action[:, 0]  
                    a_y = action[:, 1]  
                    a_h = action[:, 2]  
                    
                    #   应用输入 mask (u_t ⊙ x_t)
                    x_masked = data_x_in * a_x.view(batch_size, 1, 1).float()
                    
                    #   选择跳跃状态
                    h_skip = self.env.select_skip_hidden(h_t, a_h, H_history)
                    
                    #   前向传播
                    h_next, c_next, pred_t = self.env.Encoder(
                        x_masked, h_t, c_t, h_skip=h_skip
                    )
                    
                    pred_list.append(pred_t)
                    a_y_list.append(a_y)
                    
                    #   更新历史
                    H_history.append(h_t.detach().clone())
                    if len(H_history) > self.opt.jump_step:
                        H_history.pop(0)
                    
                    h_t = h_next
                    c_t = c_next
                
                #   根据 a_y (q_t) 计算损失 min L(θ) = min (1/||q||_1) Σ q_t · ℓ(ŷ_t, y_t)
                pred = torch.cat(pred_list, dim=1)  # (batch, T, H)
                a_y_mask = torch.stack(a_y_list, dim=1).float()  # (batch, T)
                
                loss_per_step = (pred - data_y) ** 2  # (batch, T, H)
                loss_per_step = loss_per_step.mean(dim=2)  # (batch, T)
                
                #   应用 a_y mask (只计算 q_t=1 的时间步)
                masked_loss = loss_per_step * a_y_mask
                
                #   归一化：除以选择的时间步数量
                num_selected = a_y_mask.sum() + 1e-10
                loss = torch.sqrt(masked_loss.sum() / num_selected)
                
                # 反向传播
                self.env.optimizer.zero_grad()
                loss.backward()
                self.env.optimizer.step()
            
            self.env.scheduler.step()

    
    def train_environment_without_skip(self, train_loader, iter):
        for epoch in range(iter):
            self.logger.info("environment training epoch: {}".format(epoch))
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                data_x, data_y = batch_x.detach().clone(), batch_y.detach().clone()
                batch_size = data_x.shape[0]
                batch_pred_y, batch_pred_agent =self.forward(batch_x)
                                

                self.env.reset_environment(data_x, data_y, batch_pred_agent)
            self.env.scheduler.step()
            
    def train_agent(self, train_loader, val_loader):
        """训练 agent"""
        for epoch in range(self.opt.ppo_epochs):
            self.logger.info(f"  agent training epoch: {epoch}")
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                data_x, data_y = batch_x.detach().clone(), batch_y.detach().clone()

                batch_size, T, _ = batch_x.size()
                
                # 初始化隐藏状态
                h_0 = torch.zeros(
                    self.opt.encoder_num_layer,
                    batch_size,
                    self.opt.encoder_hidden_size
                ).to(self.device)
                c_0 = torch.zeros(
                    self.opt.encoder_num_layer,
                    batch_size,
                    self.opt.encoder_hidden_size
                ).to(self.device)
                
                h_t = h_0
                c_t = c_0
                
                # H_history = [h_{t-K}, h_{t-K+1}, ..., h_{t-2}, h_{t-1}]
                H_history = []                
                ep_reward = 0                
                # 逐步预测并选择动作
                for t in range(T):
                    data_x_in = data_x[:, t, :].unsqueeze(1)  # (batch, 1, input_size)
                    
                    #构造状态: s_t = concat(h_{t-1}, x_t)
                    h_flat = h_t.permute(1, 0, 2).reshape(batch_size, -1)
                    x_flat = data_x[:, t, :]
                    state = torch.cat([h_flat, x_flat], dim=1)
                    
                    # 选择动作 [a_x, a_y, a_h]
                    action, action_log_prob = self.ppoagent.choose_action(state)
                    
                    # 环境交互（包含跳跃连接）
                    h_next, c_next, reward, ypred = self.env.observation(
                        action, h_t, data_y[:, t], data_x_in, c_t, H_history)
                                        
                    # 构造下一个状态: s_{t+1} = concat(h_t, x_{t+1})
                    if t + 1 < T:
                        x_next_flat = data_x[:, t + 1, :]
                    else:
                        x_next_flat = torch.zeros_like(x_flat).to(self.device)
                    
                    h_next_flat = h_next.permute(1, 0, 2).reshape(batch_size, -1)
                    next_state = torch.cat([h_next_flat, x_next_flat], dim=1)
                    
                    # 存储转换
                    transitions = [self.Transition(                    
                        state[b], action[b], reward[b], action_log_prob[b], next_state[b])                    
                    for b in range(batch_size)]
                                    
                    # 批量添加到缓冲区
                    self.ppoagent.buffer.extend(transitions)
                    self.ppoagent.predictions_buffer.extend(ypred.squeeze(1))
                    self.ppoagent.targets_buffer.extend(data_y[:, t])
                    self.ppoagent.counter += batch_size
                    
                    # trans = self.Transition(
                    #     state, action, reward, action_log_prob, next_state
                    # )
                    
                    # 更新历史隐藏状态（维护最近 K 个）在更新前添加当前状态
                    H_history.append(h_t.detach().clone())
                    if len(H_history) > self.opt.jump_step:
                        H_history.pop(0)  # 移除最旧的状态
                    
                    # 更新状态
                    h_t = h_next
                    c_t = c_next
                    ep_reward += reward
                    
                    # 如果 buffer 满了，执行 PPO 更新
                    # if self.ppoagent.store_transition(transitions):
                    if self.ppoagent.counter >= self.buffer_capacity:                   
                        self.ppoagent.update()
        


    def train_agent_without_action(self,train_loader, val_loader):
        
        for epoch in range(self.opt.agent_iter):
            self.logger.info("  agent training epoch: {}".format(epoch))
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)# (batch_size, seq_len, input_size)
                batch_y = batch_y.to(self.device)
                data_x, data_y = batch_x.detach().clone(), batch_y.detach().clone()
                batch_size = data_x.shape[0]
                h_0 = torch.zeros(self.opt.encoder_num_layer, batch_size, self.opt.encoder_hidden_size).to(self.device)
                h_t = h_0
                ep_reward = 0
                _, T, _ = batch_x.size()
                
                H_list = []                     
                # H_list.append(h_t)
                # predict step-by-step choose action
                for t in range(T):
                    data_x_in = data_x[:, t, :].unsqueeze(1)
                    h_flat = h_t.view(h_t.size(1),-1)
                    x_flat = data_x_in.squeeze(1)
                    state = torch.cat([h_flat, x_flat], dim=1) # (batchsize, num_layers*hidden_size + input_size)                                        
                    action,action_log_prob = self.ppoagent.choose_action(state)
                    h_next,c_next,reward,ypred_t = self.env.observation(action, h_t,data_y[:,t], data_x_in, c=None)
                                        
                    #nextstate
                    if t+1 < T:
                        x_next_flat = data_x[:, t+1, :]
                    else:
                        x_next_flat = torch.zeros_like(x_flat).to(self.device)
                    h_next_flat = h_next.view(h_next.size(1),-1)
                    next_state = torch.cat([h_next_flat, x_next_flat], dim=1)
                    
                    trans = self.Transition(state, action, reward, action_log_prob,next_state)
                    
                    h_t = h_next
                    c_t = c_next
                    ep_reward += reward

                    if self.ppoagent.store_transition(trans):
                        self.ppoagent.update()


    
    def xfit_logger(self,epoch):
        
        self.logger.info('Epoch:{};Training RMSE: encoder: {:.8f} \n Validating RMSE: encoder: {:.8f}'.format(
                epoch, self.fit_info.trmse_encoder, self.fit_info.vrmse_encoder))
    

    def evaluate_encoder(self, train_loader, val_loader):
        with torch.no_grad():
            _,y,pred_freeL,pred_action = self.predict(train_loader)
            self.fit_info.trmse_encoder = torch.sqrt(self.loss_func(y,pred_freeL))

            _,val_y,pred_freeL_val,pred_action_val = self.predict(val_loader)
            self.fit_info.vrmse_encoder = torch.sqrt(self.loss_func(val_y, pred_freeL_val))

            trmse = self.fit_info.trmse_encoder
            vrmse = self.fit_info.vrmse_encoder
            
            self.fit_info.loss_list.append(trmse)
            self.fit_info.vloss_list.append(vrmse)
            self.fit_info.t_mse_list.append(trmse**2)
            self.fit_info.v_mse_list.append(vrmse**2)
            self.fit_info.encoder_loss_list.append(self.fit_info.trmse_encoder)
            self.fit_info.encoder_vloss_list.append(self.fit_info.vrmse_encoder)
            return pred_action, pred_action_val, trmse, vrmse
        
    def predict(self,data_loader):

        x = []
        y = []
        pred_y = list()
        pred_agent = list()
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_pred_y, batch_pred_agent = self.forward(batch_x)
            x.append(batch_x)
            y.append(batch_y)
            pred_y.append(batch_pred_y)
            pred_agent.append(batch_pred_agent)
        x = torch.cat(x, dim=0).detach().cpu()
        y = torch.cat(y, dim=0).detach().cpu()
        pred_y = torch.cat(pred_y, dim=0).detach().cpu()
        pred_agent = torch.cat(pred_agent, dim=0)
        return x, y, pred_y, pred_agent
    
    def get_action(self, batch_x,batch_y):
        x = []
        y = []
        pred_y = list()
        pred_agent = list()
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_pred_y, batch_pred_agent = self.forward(batch_x)
        x.append(batch_x)
        y.append(batch_y)
        pred_y.append(batch_pred_y)
        pred_agent.append(batch_pred_agent)
        x = torch.cat(x, dim=0).detach().cpu()
        y = torch.cat(y, dim=0).detach().cpu()
        pred_y = torch.cat(pred_y, dim=0).detach().cpu()
        pred_agent = torch.cat(pred_agent, dim=0)
        return x, y, pred_y, pred_agent
    
    
    def forward(self, batch_x):
        """
        前向传播（推理时使用）        
        参数:
            batch_x: (batch_size, T, C)        
        返回:
            pred_y: (batch_size,T, H)
            pred_agent: (batch_size, T, 3) - [a_x, a_y, a_h]
        """
        pred_y = []
        pred_agent = []
        batch_size, T, C = batch_x.size()
        
        # 初始化
        h_0 = torch.zeros(
            self.opt.encoder_num_layer,
            batch_size,
            self.opt.encoder_hidden_size
        ).to(self.device)
        c_0 = torch.zeros(
            self.opt.encoder_num_layer,
            batch_size,
            self.opt.encoder_hidden_size
        ).to(self.device)
        
        h_t = h_0
        c_t = c_0
        H_history = []
        
        for t in range(T):
            batch_x_in = batch_x[:, t, :].unsqueeze(1)
            
            #   构造状态
            h_flat = h_t.permute(1, 0, 2).reshape(batch_size, -1)
            x_flat = batch_x[:, t, :]
            state = torch.cat([h_flat, x_flat], dim=1)
            
            #   选择动作
            action, _ = self.ppoagent.choose_action(state)
            pred_agent.append(action.unsqueeze(1))
            
            #   提取动作
            a_x = action[:, 0]
            a_y = action[:, 1]
            a_h = action[:, 2]
            
            #   应用动作
            x_masked = batch_x_in * a_x.view(batch_size, 1, 1).float()
            h_skip = self.env.select_skip_hidden(h_t, a_h, H_history)
            
            #   前向传播
            h_next, c_next, pred_t = self.env.Encoder(
                x_masked, h_t, c_t, h_skip=h_skip
            )
            
            pred_y.append(pred_t)
            
            #   更新历史
            H_history.append(h_t.detach().clone())
            if len(H_history) > self.opt.jump_step:
                H_history.pop(0)
            
            h_t = h_next
            c_t = c_next
        
        pred_y = torch.cat(pred_y, dim=1)
        pred_agents = torch.cat(pred_agent, dim=1)
        
        return pred_y, pred_agents

    
    def forward_without_skip(self, batch_x):
        pred_y = []
        pred_agent = []
        batch_size,L,C = batch_x.size()
        h_0 = torch.zeros(self.opt.encoder_num_layer, batch_size, self.opt.encoder_hidden_size).to(self.device)
        h_t = h_0
        #self.environment.Encoder.load_state_dict(torch.load('initial_rnn_model_params.pth'))
        for t in range(self.T):
            batch_x_in = batch_x[:, t, :].unsqueeze(1)
            h_flat = h_t.view(h_t.size(1),-1)
            x_flat = batch_x_in.squeeze(1)
            state = torch.cat([h_flat, x_flat], dim=1) # (batch

            action,_ = self.ppoagent.choose_action(state)
            a_tensor = action.unsqueeze(1)
            pred_agent.append(a_tensor)
            h_next, c_next, pred_t = self.env.Encoder(batch_x_in, h_t, None,m=action)
            pred_y.append(pred_t)
            h_t = h_next
            c_t = c_next
        pred_y = torch.cat(pred_y, dim=1)
        pred_agents = torch.cat(pred_agent, dim=1)
        return pred_y, pred_agents
    def predict_with_mdp(self,data_loader, mdp_list):
        x = []
        y = []
        pred_y = list()
        pred_agent = list()
        b_s, _, _ = concentrateLoader(data_loader, self.device)
        #mdp_list = mdp_list[:b_s, :, :]
        #window_size = 512
        i = 0
        for batch_x, batch_y in data_loader:
            #start = i * window_size
            #end = min((i + 1) * window_size, len(mdp_list))
            #mdp_lists = mdp_list[start:end, :, :]
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_size, _, _ = batch_x.shape
            #mdp_list = mdp_list[:batch_size, :, :]
            batch_pred_y, batch_pred_agent = self.forward_with_mdp(batch_x, mdp_list)
            i = i + 1
            x.append(batch_x)
            y.append(batch_y)
            pred_y.append(batch_pred_y)
            pred_agent.append(batch_pred_agent)
        x = torch.cat(x, dim=0).detach().cpu()
        y = torch.cat(y, dim=0).detach().cpu()
        pred_y = torch.cat(pred_y, dim=0).detach().cpu()
        pred_agent = torch.cat(pred_agent, dim=0)
        return x, y, pred_y, pred_agent

    def forward_with_mdp(self, batch_x, mdp_list):
        pred_y = []
        pred_agent = []
        batch_size,T,C = batch_x.size()
        h_0 = torch.zeros(self.opt.encoder_num_layer, batch_size, self.opt.encoder_hidden_size).to(self.device)
        h_t = h_0
        #self.environment.Encoder.load_state_dict(torch.load('initial_rnn_model_params.pth'))
        for t in range(T):
            batch_x_in = batch_x[:, t, :].unsqueeze(1)
            #obs = h_t.view(h_t.size(1),-1)
            #action = self.agent.select_action(obs,t)
            action = mdp_list[:, t, :].squeeze(1)
            a_tensor = action.unsqueeze(1)
            pred_agent.append(a_tensor)
            h_next, c_next, pred_t = self.env.Encoder(batch_x_in, h_t, None,m=action)
            pred_y.append(pred_t)
            h_t = h_next
            c_t = c_next
        pred_y = torch.cat(pred_y, dim=1)
        pred_agents = torch.cat(pred_agent, dim=1)
        return pred_y, pred_agents
    
    def loader_pred(self, data_loader, usiing_best= True, return_action = True):
        if self.best_environment != None and usiing_best:
            self.env.load_state_dict(self.best_environment)
            self.ppoagent.critic_net.load_state_dict(self.best_critic)
            self.ppoagent.actor_net.load_state_dict(self.best_actor)
        best_MDPLIST = self.best_mdplist
        x, y, pred_real, pred_action = self.predict(data_loader)#best_MDPLIST
        pred_last = pred_real[:, -1,:]
        y = y[:, -1,:]
        x, y, pred_last, pred_action = x.detach().cpu().numpy(), y.detach().cpu().numpy(), pred_last.detach().cpu().numpy(), pred_action.detach().cpu().numpy()
        return x, y, pred_last, pred_action

def concentrateLoader(data_loader,device):
    x = []
    y = []
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        x.append(batch_x)
        y.append(batch_y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    b, l, c = x.size()
    return b, l, c


    





            





