import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from task.TaskLoader import Opt
from tqdm import tqdm
import os
import copy
import gc
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import namedtuple
import numpy as np


class DynamicTransitionSampler:
    """
    动态转换采样器 - 实现论文中的 DTS 机制
    """
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        
        # 超参数
        self.alpha = getattr(opt, 'reward_alpha', 1.0)  # 预测误差敏感系数
        self.beta = getattr(opt, 'priority_beta', 0.5)  # TD误差和预测误差的权重 [0,1]
        self.lambda_min = getattr(opt, 'lambda_min', 0.1)  # 最小温度
        self.lambda_max = getattr(opt, 'lambda_max', 2.0)  # 最大温度
        self.mu = getattr(opt, 'temp_oscillation', 0.1)  # 温度振荡幅度
        self.omega = getattr(opt, 'temp_frequency', 2.0)  # 温度周期频率
        
        
        self.total_epochs = opt.agent_iter
        
    def compute_td_error(self, transitions, critic_net, gamma=0.99):
        """
        计算 TD 误差: δ_t = r_t + γ·v_φ(s_{t+1}) - v_φ(s_t)
        
        参数:
            transitions: 转换列表
            critic_net: 价值网络
            gamma: 折扣因子
        
        返回:
            td_errors: (M,) TD 误差的绝对值
        """
        states = torch.stack([t.state for t in transitions])  # (M, state_dim)
        next_states = torch.stack([t.next_state for t in transitions])
        rewards = torch.stack([t.reward for t in transitions]).squeeze(-1)  # (M,)
        
        with torch.no_grad():
            current_values = critic_net(states).squeeze(-1)  # (M,)
            next_values = critic_net(next_states).squeeze(-1)  # (M,)
            
            # TD 误差
            td_errors = rewards + gamma * next_values - current_values
            td_errors_abs = torch.abs(td_errors)  # |δ_t|
        
        return td_errors_abs
    
    def compute_forecasting_error(self, transitions, predictions, targets):
        """
        计算预测误差度量: ε_t = 1 - α/(α + ||ŷ_t - y_t||_1)
        
        参数:
            transitions: 转换列表
            predictions: 预测值列表 (M, H)
            targets: 真实值列表 (M, H)
        
        返回:
            forecasting_errors: (M,) 预测误差度量
        """
        # 计算 L1 误差
        l1_errors = torch.abs(predictions - targets).mean(dim=1)  # (M,)
        
        # 应用变换: ε_t = 1 - α/(α + ε_t)
        forecasting_metric = 1 - self.alpha / (self.alpha + l1_errors)
        
        return forecasting_metric
    
    def compute_priority_scores(self, td_errors, forecasting_errors):
        """
        计算统一优先级分数: p_t = β·|δ_t|/δ_max + (1-β)·ε_t
        
        参数:
            td_errors: (M,) TD 误差
            forecasting_errors: (M,) 预测误差度量
        
        返回:
            priorities: (M,) 优先级分数
        """
        # 归一化 TD 误差到 [0, 1]
        delta_max = td_errors.max() + 1e-10
        normalized_td = td_errors / delta_max
        
        # 组合优先级
        priorities = self.beta * normalized_td + (1 - self.beta) * forecasting_errors
        
        return priorities
    
    def compute_base_temperature(self, priorities):
        """
        计算基础温度: λ_m = λ_min + p_m·(λ_max - λ_min)
        
        参数:
            priorities: (M,) 优先级分数
        
        返回:
            base_temps: (M,) 基础温度
        """
        base_temps = self.lambda_min + priorities * (self.lambda_max - self.lambda_min)
        return base_temps
    
    def compute_adaptive_temperature(self, base_temps, epoch):
        """
        计算自适应温度（带周期性退火）:
        λ_{g,m} = λ_m · (λ_min/λ_m)^{g/G_π} · [1 + μ·sin(2πω·g/G_π)]
        
        参数:
            base_temps: (M,) 基础温度
            epoch: 当前训练轮次
        
        返回:
            adaptive_temps: (M,) 自适应温度
        """
        g = epoch
        G = self.total_epochs
        
        # 指数衰减项
        decay_factor = (self.lambda_min / base_temps) ** (g / G)
        
        angle = torch.tensor(2 * np.pi * self.omega * g / G, 
                        dtype=base_temps.dtype, 
                        device=base_temps.device)
        oscillation = 1 + self.mu * torch.sin(angle)
                
        adaptive_temps = base_temps * decay_factor * oscillation
        
        return adaptive_temps
    
    def compute_sampling_probabilities(self, priorities, temperatures):
        """
        计算采样概率（温度缩放的 softmax）:
        p̃_{g,m} = exp(p_m/λ_{g,m}) / Σ_j exp(p_j/λ_{g,j})
        
        参数:
            priorities: (M,) 优先级分数
            temperatures: (M,) 自适应温度
        
        返回:
            probs: (M,) 采样概率
        """
        # 温度缩放
        scaled_priorities = priorities / temperatures
        
        # Softmax
        probs = F.softmax(scaled_priorities, dim=0)
        
        return probs
    
    def sample_transitions(self, buffer, num_samples, critic_net,current_epoch,predictions=None, targets=None):
        """
        从经验缓冲区中采样转换
        
        参数:
            buffer: 经验缓冲区（转换列表）
            num_samples: 采样数量
            critic_net: 价值网络
            predictions: 预测值 (M, H)
            targets: 真实值 (M, H)
        
        返回:
            sampled_indices: 采样的索引
            sampled_transitions: 采样的转换
        """
        N = len(buffer)
        
        # 1. 计算 TD 误差
        td_errors = self.compute_td_error(buffer, critic_net, self.opt.gamma)
        
        # 2. 计算预测误差（如果提供）
        if predictions is not None and targets is not None:
            forecasting_errors = self.compute_forecasting_error(
                buffer, predictions, targets
            )
        else:
            # 如果没有预测误差，只使用 TD 误差
            forecasting_errors = torch.zeros_like(td_errors)
        
        # 3. 计算优先级分数
        priorities = self.compute_priority_scores(td_errors, forecasting_errors)
        
        # 4. 计算基础温度
        base_temps = self.compute_base_temperature(priorities)
        
        # 5. 计算自适应温度
        adaptive_temps = self.compute_adaptive_temperature(
            base_temps, current_epoch
        )
        
        # 6. 计算采样概率
        sampling_probs = self.compute_sampling_probabilities(
            priorities, adaptive_temps
        )
        
        # 7. 根据概率采样
        sampled_indices = torch.multinomial(
            sampling_probs, 
            num_samples=min(num_samples, N),
            replacement=False
        )
        
        sampled_transitions = [buffer[idx] for idx in sampled_indices]
        
        return sampled_indices, sampled_transitions