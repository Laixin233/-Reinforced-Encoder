import torch
import torch.nn as nn
from models.RNNs.maskRNN import MaskJumpEncoder,Encoder
from tqdm import tqdm

class Environment(nn.Module):
    def __init__(self, opts=None, logger=None):
        super().__init__()
        self.opts = opts
        self.logger = logger

        self.component = opts.component
        self.reward_alpha = getattr(opts, 'reward_alpha', 1.0)  # 奖励敏感系数
        self.reward_threshold = getattr(opts, 'reward_threshold', 0.5)  # 阈值 c
        
        self.device = opts.device
        self.input_size = opts.num_variate
        self.d = opts.patch_size
        self.hidden_size = opts.encoder_hidden_size
        self.num_layer = opts.encoder_num_layer
        if self.opts.if_patch:
            self.input_size = opts.patch_size
        else:
            self.input_size = opts.num_variate


        self.T = opts.steps
        self.jump_step = opts.jump_step
        self.output_size = opts.H 
        # self.batch_size = opts.batch_size
        # self.h_0 = torch.zeros(self.num_layer, self.batch_size, self.hidden_size)


        self.environment_lr = opts.learning_rate
        self.environment_gamma = opts.step_gamma

        self.Encoder = MaskJumpEncoder(self.component, self.input_size, self.hidden_size, self.num_layer,self.output_size, self.device)
        self.loss_fn = nn.MSELoss()
        paras = self.Encoder.parameters()
        if self.opts.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(paras, lr=self.environment_lr)#,weight_decay=0.0001
        elif self.opts.optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(paras, lr=self.environment_lr,weight_decay=0.001)
        elif self.opts.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(paras, lr=self.environment_lr,weight_decay=0.001)
        else:
            self.optimizer = torch.optim.SGD(paras, lr=self.environment_lr,weight_decay=0.001)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.environment_gamma)
        self.done = False

    # def initial_environment(self, train_loader, val_loader,preTrain_epochs):
    #     self.logger.info('PreTrain seq2seq....')
    #     m_t = torch.ones(1,3).to(self.device)
    #     self.train_rnn(train_loader,preTrain_epochs,m_t)
    #     # loss_list = self.preEvaluation(train_loader)
    #     vloss_list = self.validate_rnn(val_loader,m=m_t)
    #     return vloss_list
    
    # def train_rnn(self,train_loader,preTrain_epochs,m = None):
    #     for _ in tqdm(range(preTrain_epochs)):
    #         for batch_x, batch_y in train_loader:
    #             batch_x = batch_x.to(self.device).float()
    #             batch_y = batch_y.to(self.device).float()
    #             data_x, data_y = batch_x.detach().clone(), batch_y.detach().clone()
    #             B,T,C = data_x.size()
    #             H_list = []

    #             pred = list()
                
    #             h_t = torch.zeros(self.num_layer, B, self.hidden_size).to(self.device)
    #             H_list.append(h_t)
    #             c_t = torch.zeros(self.num_layer, B, self.hidden_size).to(self.device)

    #             # predict step-by-step
    #             for t in range(T):
    #                 input_x = data_x[:,t,:].unsqueeze(1)
    #                 if t < self.jump_step:
    #                     # h_t = torch.zeros(self.num_layer, B, self.hidden_size).to(self.device)
    #                     h_t = H_list[-1]
    #                     h_t,c_t,pred_t = self.Encoder(input_x,h_t,c_t,m)
    #                 else:
    #                     h_t = H_list[t - self.jump_step]
    #                     h_t,c_t,pred_t = self.Encoder(input_x,h_t,c_t,m)
    #                 H_list.append(h_t)
    #                 pred.append(pred_t)
    #             pred = torch.cat(pred,dim = 1)
    #             loss = torch.sqrt(self.loss_fn(pred,data_y))
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #         self.scheduler.step()

    #     self.logger.info('PreTrain seq2seq Finish.')
    #     print('train loss:',loss.item()) 
        

    # def validate_rnn(self,val_loader,m = None):
    #     loss_list = list()
    #     with torch.no_grad():
    #         for batch_x, batch_y in val_loader:
    #             batch_x = batch_x.to(self.device)
    #             batch_y = batch_y.to(self.device)
    #             data_x, data_y = batch_x.detach().clone(), batch_y.detach().clone()
    #             B,T,C = data_x.size()
    #             H_list = []

    #             h_t = torch.zeros(self.num_layer, B, self.hidden_size).to(self.device)
    #             c_t = torch.zeros(self.num_layer, B, self.hidden_size).to(self.device)
    #             H_list.append(h_t)
                
    #             pred = list()
    #             # predict step-by-step
    #             for t in range(T):
    #                 input_x = data_x[:,t,:].unsqueeze(1)
    #                 if t < self.jump_step:
    #                     # h_t = torch.zeros(self.num_layer, B, self.hidden_size).to(self.device)
    #                     h_t = H_list[-1]
    #                     h_t,c_t,pred_t = self.Encoder(input_x,h_t,c_t,m)
    #                 else:
    #                     h_t = H_list[t - self.jump_step]
    #                     h_t,c_t,pred_t = self.Encoder(input_x,h_t,c_t,m)
    #                 H_list.append(h_t)
    #                 pred.append(pred_t)
    #             pred = torch.cat(pred,dim = 1)
    #             loss = torch.sqrt(self.loss_fn(pred,data_y))
    #             loss_list.append(loss.item())
    #     return loss_list
    def initial_environment(self, train_loader, val_loader, preTrain_epochs):
        """
        初始化环境 - 预训练 Encoder（不使用跳跃连接）
        """
        self.logger.info('PreTrain Encoder (without actions)...')
        self.train_rnn(train_loader, preTrain_epochs)
        vloss_list = self.validate_rnn(val_loader)
        return vloss_list
    
    def train_rnn(self, train_loader, preTrain_epochs):
        """
        预训练 RNN - 使用标准 RNN 前向传播
        """
        for epoch in tqdm(range(preTrain_epochs), desc="PreTraining"):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                data_x, data_y = batch_x.detach().clone(), batch_y.detach().clone()
                B, T, C = data_x.size()

                pred = []
                
                # 初始化隐藏状态
                h_t = torch.zeros(self.num_layer, B, self.hidden_size).to(self.device)
                c_t = torch.zeros(self.num_layer, B, self.hidden_size).to(self.device)

                #   逐步预测（不使用跳跃连接）
                for t in range(T):
                    input_x = data_x[:, t, :].unsqueeze(1)  # (B, 1, C)
                    
                    #   前向传播（h_skip=None 表示不使用跳跃连接）
                    h_t, c_t, pred_t = self.Encoder(input_x, h_t, c_t, h_skip=None)
                    pred.append(pred_t)
                
                # 计算损失
                pred = torch.cat(pred, dim=1)  # (B, T, H)
                loss = torch.sqrt(self.loss_fn(pred, data_y))
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            self.scheduler.step()
            avg_loss = epoch_loss / num_batches
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{preTrain_epochs}], Loss: {avg_loss:.6f}')

        self.logger.info('PreTrain Encoder Finish.')
        self.logger.info(f'Final training loss: {avg_loss:.6f}')

    def validate_rnn(self, val_loader):
        """
        验证 RNN - 不使用跳跃连接
        """

        loss_list = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                data_x, data_y = batch_x.detach().clone(), batch_y.detach().clone()
                B, T, C = data_x.size()

                pred = []
                
                # 初始化隐藏状态
                h_t = torch.zeros(self.num_layer, B, self.hidden_size).to(self.device)
                c_t = torch.zeros(self.num_layer, B, self.hidden_size).to(self.device)
                
                #   逐步预测（不使用跳跃连接）
                for t in range(T):
                    input_x = data_x[:, t, :].unsqueeze(1)
                    
                    # 前向传播（h_skip=None）
                    h_t, c_t, pred_t = self.Encoder(input_x, h_t, c_t, h_skip=None)
                    pred.append(pred_t)
                
                # 计算损失
                pred = torch.cat(pred, dim=1)
                loss = torch.sqrt(self.loss_fn(pred, data_y))
                loss_list.append(loss.item())
        
        avg_vloss = sum(loss_list) / len(loss_list)
        self.logger.info(f'Validation Loss: {avg_vloss:.6f}')
        
        return loss_list


    def train_encoder(self, data_loader, preTrain_epochs,m_t):
        self.logger.info('PreTrain encoder...')
        for t in tqdm(range(preTrain_epochs)):
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                data_x, data_y = batch_x.detach().clone(), batch_y.detach().clone()

                pred = list()

                a_x = torch.tensor([1])
                a_h = torch.tensor([1])
                a_y = torch.tensor([1])
                if t == 0:
                    h_next, c_next, pred_t = self.Encoder(data_x, None, None, a_x, a_h, a_y)
                    h_t = h_next
                    c_t = c_next
                    pred.append(pred_t)

                else:
                    h_next, c_next, pred_t = self.Encoder(data_x, h_t, None, a_x, a_h, a_y)
                    h_t = h_next
                    c_t = c_next
                    pred.append(pred_t)
                pred = torch.cat(pred, dim=-1)
                loss = torch.sqrt(self.loss_fn(pred,data_y))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        self.logger.info('PreTrain seq2seq Finish.')
        torch.save(self.Encoder.state_dict(), 'initial_rnn_model_params.pth')
    def preEvaluation(self, data_loader):
        #loss_list = {}
        _,y,pred = self.loader_pred(data_loader)
        loss_list = torch.sqrt(self.loss_fn(pred,y))
        return loss_list
    
    def loader_pred(self, data_loader):
        x = []
        y = []
        pred = []
        a_x = torch.tensor([1])
        a_h = torch.tensor([1])
        a_y = torch.tensor([1])
        for batch_x, batch_y in tqdm (data_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            _,L,_ = batch_x.size(1)
            for t in range(L):
                batch_x_in = batch_x[:, t, :]
                if t == 0:
                    h_next, c_next, v_pred = self.Encoder(batch_x_in, self.h_0, None, a_x, a_h, a_y)
                    h_t = h_next
                    c_t = c_next
                else:
                    h_next, c_next, v_pred = self.Encoder(batch_x_in, h_t, None, a_x, a_h, a_y)
                    h_t = h_next
                    c_t = c_next
            pred.append(v_pred)
            x.append(batch_x)
            y.append(batch_y)
            pred.append(v_pred)
        x = torch.cat(x, dim=0).detach().cpu().numpy()
        y = torch.cat(x, dim=0).detach().cpu().numpy()
        pred = torch.cat(pred, dim=0).detach().cpu().numpy()
        loss = nn.sqrt(nn.MSELoss(pred, y))
        return loss

    def observation_without_skip(self, action, h_t,y_t, x, c):
        action_x,action_y,action_h = action[:,0],action[:,1],action[:,2]

        with torch.no_grad():
            h_next, c_next, ypred_t = self.Encoder(x, h_t, c, m=action)
            reward = self.reward_func(ypred_t,action_y,y_t) #(b,h)
        return h_next, c_next, reward,ypred_t
    
    def observation(self, action, h_t, y_t, x_t, c_t, H_history):
        """
        环境观测函数 - 执行动作并返回奖励        
        参数:
            action: (batch_size, 3) - [a_x, a_y, a_h]
            h_t: 当前隐藏状态 (num_layers, batch_size, hidden_size)
            y_t: 真实标签 (batch_size, H)
            x_t: 当前输入 (batch_size, 1, input_size)
            c_t: cell state (num_layers, batch_size, hidden_size)
            H_history: 历史隐藏状态列表 [h_{t-K}, ..., h_{t-2}, h_{t-1}]        
        返回:
            h_next: 下一个隐藏状态
            c_next: 下一个 cell state
            reward: 奖励
            ypred_t: 预测输出
        """
        batch_size = x_t.size(0)
        

        a_x = action[:, 0]  # 输入特征选择 (batch_size,)
        a_y = action[:, 1]  # 输出目标选择 (batch_size,) 
        a_h = action[:, 2]  # 跳跃连接选择 (batch_size,)
        
        with torch.no_grad():
            x_masked = x_t * a_x.view(batch_size, 1, 1).float()
            h_skip = self.select_skip_hidden(h_t, a_h, H_history)
            h_next, c_next, ypred_t = self.Encoder(
                x_masked, h_t, c_t, h_skip=h_skip
            )
            reward = self.reward_func(ypred_t, a_y, y_t)
        
        return h_next, c_next, reward, ypred_t
    
    def select_skip_hidden(self, h_t, a_h, H_history):
        """
        根据动作 a_h (k_t) 选择跳跃连接的隐藏状态        
        参数:
            h_t: 当前隐藏状态 (num_layers, batch_size, hidden_size)
            a_h: 跳跃动作 (batch_size,) - 值在 [0, K]
            H_history: 历史隐藏状态列表，长度最多为 K
                    [h_{t-K}, h_{t-K+1}, ..., h_{t-2}, h_{t-1}]        
        返回:
            h_skip: 选择的跳跃隐藏状态 (num_layers, batch_size, hidden_size)
        """
        batch_size = h_t.size(1)
        num_layers = h_t.size(0)
        hidden_size = h_t.size(2)
        
        zero_tensor = torch.zeros_like(h_t).to(self.device)
        
        # 返回零向量
        if len(H_history) == 0:
            return zero_tensor
                
        # H_tensor: (history_len, num_layers, batch_size, hidden_size)
        H_tensor = torch.stack(H_history, dim=0)
        history_len = H_tensor.size(0)        

        batch_indices = torch.arange(batch_size, device=self.device)
        
        # H_history[-k] 对应 h_{t-k}
        a_h_long = a_h.long()
        indices = history_len - a_h_long  # (batch_size,)
        
        # （a_h > history_len ）
        selected_indices = torch.clamp(indices, min=0, max=history_len - 1)
        
        # H_tensor: (history_len, num_layers, batch_size, hidden_size)
        #  h_skip[:, b, :] = H_tensor[indices[b], :, b, :]
        h_skip = H_tensor[selected_indices, :, batch_indices, :]  # (batch_size, num_layers, hidden_size)
        h_skip = h_skip.permute(1, 0, 2).contiguous()  # (num_layers, batch_size, hidden_size)
        
        #  a_h = 0 的情况（non-skip）

        invalid_mask = (a_h_long == 0) | (a_h_long > history_len)
        h_skip[:, invalid_mask, :] = zero_tensor[:, invalid_mask, :]
        return h_skip
        
        
    
    # def reward_func(self,ypred_t,a_y,y_t):
    #     #要保证输入都是一样的维度
    #     a_y = a_y.unsqueeze(1).repeat(1,y_t.shape[1])
    #     reward = 0.1/(0.1+torch.abs(ypred_t.squeeze(1)- a_y*y_t))
    #     reward = torch.mean(reward,dim=1).unsqueeze(1)
    #     return reward
    
    def reward_func(self, ypred_t, a_y, y_t):
        """
        计算奖励函数
        
        参数:
            ypred_t: 预测值，形状 (batch_size, 1, H) 或 (batch_size, H)
            a_y: 动作（是否输出），形状 (batch_size,)
            y_t: 真实值，形状 (batch_size, H)
        
        返回:
            reward: 奖励值，形状 (batch_size, 1)
        """
        # 确保维度一致 ypred_t(batch_size, 1,H), a_y(batch_size,), y_t(batch_size,H)
        if len(ypred_t.shape) == 3:
            ypred_t = ypred_t.squeeze(1)  # (batch_size, H)
        
        #  ||ŷ_t - y_t||_1
        maeloss = torch.abs(ypred_t - y_t).mean(dim=1) # (batch_size,H)
        
        r_mae = self.reward_alpha / (self.reward_alpha + maeloss)
        r_mae_mean = torch.mean(r_mae)       
        #  α / (α + MAE) - c
        reward_core = self.reward_alpha / (self.reward_alpha + maeloss) - self.reward_threshold
        # reward_core = r_mae - r_mae_mean
        
        #  乘以动作系数 a_y, a_y 为 1 表示输出，为 0 表示不输出
        a_y_float = a_y.float()  # (batch_size,)
        
        # 方案1: 直接乘以动作
        reward = a_y_float * reward_core
        
        # 方案2: 如果不输出(a_y=0)，给予小的负奖励
        # reward = torch.where(a_y.bool(), 
        #                      reward_core, 
        #                      torch.full_like(reward_core, -0.1))
        
        return reward.unsqueeze(1)  # (batch_size, 1)
    
    def reset_environment(self, data_x, data_y, MDP_lists):
        batch_size,T,_ = data_x.shape
        t = 0
        pred = list()
        h0 = torch.zeros(self.opts.encoder_num_layer, batch_size, self.opts.encoder_hidden_size).to(self.device)
        for t in range(T):
            m = MDP_lists[:, t, :].squeeze(1)
            input_data = data_x[:, t, :].unsqueeze(1)
            if t == 0:
                h_t, c_t, output_t = self.Encoder(input_data, h0, None, m)
            else:
                h_t, c_t, output_t = self.Encoder(input_data, h_t, c_t,m)
            pred.append(output_t)
        pred = torch.cat(pred, dim = 1).squeeze(2)
        loss = torch.sqrt(self.loss_fn(pred, data_y))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train_environment(self, data_x, data_y, MDP_lists):
        batch_size,L,_ = data_x.shape
        t = 0
        pred = list()
        H_list = list()
        h0 = torch.zeros(self.opts.encoder_num_layer, batch_size, self.opts.encoder_hidden_size).to(self.device)
        c_0 = torch.zeros(self.opts.encoder_num_layer, batch_size, self.opts.encoder_hidden_size).to(self.device)
        H_list.append(h0)
        C_list = [c_0]
        h_t = h0
        c_t = c_0
        for t in range(L):
            m = MDP_lists[:, t, :].squeeze(1)
            input_data = data_x[:, t, :].unsqueeze(1)
            action_h = m[:,-1].long()
            if t < self.jump_step:
                h_t = H_list[-1]
                c_t = C_list[-1]
            else:
                h_t = H_tensor[action_h,:,torch.arange(batch_size),:].permute(1,0,2).contiguous()#num,num_layer,batch,hidden
                c_t = C_tensor[action_h,:,torch.arange(batch_size),:].permute(1,0,2).contiguous()
            h_t, c_t, output_t = self.Encoder(input_data, h_t, c_t,m)
            if c_t is None:
                c_t = torch.zeros(self.opts.encoder_num_layer, batch_size, self.opts.encoder_hidden_size).to(self.device)
            H_list.append(h_t)
            C_list.append(c_t)            
            H_tensor = torch.stack(H_list[-self.jump_step:],dim=0)
            C_tensor = torch.stack(C_list[-self.jump_step:],dim=0)

            pred.append(output_t)
        pred = torch.cat(pred, dim = 1).squeeze(2)
        loss = torch.sqrt(self.loss_fn(pred, data_y))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    

    def objective_function(self, mdp_list,train_loader, val_loader,preTrain_epochs):
        self.logger.info('PreTrain seq2seq....')

        self.train_process(train_loader,preTrain_epochs,mdp_list)
        # loss_list = self.preEvaluation(train_loader)
        score = self.validate_process(val_loader,mdp_list)
        return score
    
    def train_process(self,train_loader,preTrain_epochs,mdp_list = None):
        for _ in tqdm(range(preTrain_epochs)):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                data_x, data_y = batch_x.detach().clone(), batch_y.detach().clone()

                pred = list()
                # predict step-by-step
                for t in range(self.T):
                    input_x = data_x[:,t,:].unsqueeze(1)
                    m = mdp_list[:,t,:]
                    if t== 0:
                        h_t,c_t,pred_t = self.Encoder(input_x,m=m)
                    else:
                        h_t,c_t,pred_t = self.Encoder(input_x,h_t,c_t,m)
                    pred.append(pred_t)
                pred = torch.cat(pred,dim = 1)
                loss = torch.sqrt(self.loss_fn(pred,data_y))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        self.logger.info('PreTrain seq2seq Finish.')
    
    def validate_process(self,val_loader,mdp_list = None):
        loss_list = list()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                data_x, data_y = batch_x.detach().clone(), batch_y.detach().clone()
                
                pred = list()
                # predict step-by-step
                for t in range(self.T):
                    input_x = data_x[:,t,:].unsqueeze(1)
                    m = mdp_list[:,t,:]
                    if t == self.T:
                        m[:,2] = 1
                    if t== 0:
                        h_t,c_t,pred_t = self.Encoder(input_x,m=m)
                    else:
                        h_t,c_t,pred_t = self.Encoder(input_x,h_t,c_t,m)
                    pred.append(pred_t)
                pred = torch.cat(pred,dim = 1)
                loss = torch.sqrt(self.loss_fn(pred,data_y))
                loss_list.append(loss.item())
                score = self.score_func(pred[:,:self.T,:],data_y[:,:self.T,:])
        return score
    
    def score_func(self,y_true,y_pred):
        loss = torch.sqrt(self.loss_fn(y_pred,y_true))
        reward = 0.1/(0.1+loss)
        # reward = torch.mean(reward,dim=1).unsqueeze(1)
        return reward
    
    def test_predict(self,val_loader,mdp_list = None):

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                data_x, data_y = batch_x.detach().clone(), batch_y.detach().clone()
                
                pred = list()
                # predict step-by-step
                for t in range(self.T):
                    input_x = data_x[:,t,:].unsqueeze(1)
                    m = mdp_list[:,t,:]
                    if t == self.T:
                        m[:,2] = 1
                    if t== 0:
                        h_t,c_t,pred_t = self.Encoder(input_x,m=m)
                    else:
                        h_t,c_t,pred_t = self.Encoder(input_x,h_t,c_t,m)
                    pred.append(pred_t)
                pred = torch.cat(pred,dim = 1)
            
    
