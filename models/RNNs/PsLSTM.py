#P-sLSTM: Unlocking the Power of LSTM for Long Term Time Series Forecasting

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from  PyxLSTM.xLSTM import xLSTM

class Patching(nn.Module):
    def __init__(self, patch_size, stride):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, x):
        return x.unfold(dimension=-1, size=self.patch_size, step=self.stride)

class ChannelIndependence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return rearrange(x, 'b c l -> (b c) l')

class Embedding(nn.Module):
    def __init__(self, patch_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Linear(patch_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)
    
class Attention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(0.1)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        # self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        # Split heads
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhnl,bhnd->bhnd", attn, v)

        return out,attn

class DQNxLSTM(nn.Module):
    def __init__(self,embedding_dim, hidden_size, num_layers,num_blocks=1):
        super().__init__()

        self.xlstm = xLSTM(embedding_dim, hidden_size, num_layers,num_blocks=num_blocks)


    def forward(self,x):
        B, L, C = x.shape
        

        h_t = torch.zeros(2, B, self.xlstm.hidden_size).to(x.device)
        c_t = torch.zeros(2, B, self.xlstm.hidden_size).to(x.device)

        for i in range(L):
            # a
            # r
            out,(h_t,c_t) = self.xlstm(x,(h_t,c_t))
        return out


class sLSTM_block(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers, num_heads, conv1d_kernel_size):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.conv1d_kernel_size = conv1d_kernel_size

        # Multi-head Attention
        self.attentionlayer = Attention(embedding_dim, num_heads)

        self.dropout = nn.Dropout(0.1)

        # Conv1d
        self.conv1d = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=conv1d_kernel_size, padding='same')

        # State Space Model
        self.ssm = nn.Linear(embedding_dim, embedding_dim)

        # xlstm
        # self.xlstm = xLSTM(embedding_dim, hidden_size, num_layers,num_blocks=1 )
        self.dqnxlstm = DQNxLSTM(embedding_dim, hidden_size, num_layers,num_blocks=1)

        # Output Projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # Input Normalization
        x = self.norm(x)

        # Multi-head Attention
        context,attn = self.attentionlayer(x,x,x)
        # Merge heads
        context = rearrange(context, 'b h n d -> b n (h d)')
        # Context
        context = self.dqnxlstm(context)

        # Conv1d
        context = self.conv1d(context.permute(0, 2, 1)).permute(0, 2, 1)

        # State Space Model
        context = self.ssm(context)

        # Output Projection
        context = self.out_proj(context)

        # Residual Connection
        context = context + x

        return context

class xLSTMblock(nn.Module):
    def __init__(self, embedding_dim, num_heads, conv1d_kernel_size, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([xLSTM(embedding_dim, num_heads, conv1d_kernel_size) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class Projection(nn.Module):
    def __init__(self, embedding_dim, pred_len):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, pred_len)

    def forward(self, x):
        return self.proj(x)

class Reshape(nn.Module):
    def __init__(self, B, C):
        super().__init__()
        self.B = B
        self.C = C

    def forward(self, x):
        x = rearrange(x, '(b c) l -> b c l', b=self.B, c=self.C)
        x = rearrange(x, 'b c l -> b l c')
        return x

class P_sLSTM(nn.Module):
    def __init__(self, seq_len, pred_len, channel,hidden_size, num_layers, embedding_dim, patch_size, stride, conv1d_kernel_size, num_heads, num_blocks):
        super().__init__()
        self.patching = Patching(patch_size, stride)
        self.channel_independence = ChannelIndependence()
        self.embedding = Embedding(patch_size, embedding_dim)
        self.slstmblock = sLSTM_block(embedding_dim, hidden_size, num_layers, num_heads, conv1d_kernel_size)
        self.flatten = nn.Flatten(1)
        self.projection = Projection(embedding_dim, pred_len)
        self.reshape = Reshape(1, pred_len)

    def forward(self, x):
        # Input: [B, L, C]
        B, L, C = x.shape
        x = rearrange(x, 'b l c -> b c l')  # [B, C, L]
        x = rearrange(x, 'b c l -> (b c) l')  # [(B*C), L]
        x = self.patching(x)  # [(B*C), Num_patches, Patch_size]
        x = self.embedding(x)  # [(B*C), Num_patches, Embedding_dim]
        x = self.slstmblock(x)  # [(B*C), Num_patches, Embedding_dim]
        # x = self.flatten(x)  # [(B*C), Num_patches * Embedding_dim]
        x = self.projection(x)  # [(B*C), Num_patches, Embedding_dim] -> [(B*C), Pred_len]
        x = self.reshape(x)  # [B, Pred_len, C]
        return x

# Example usage
if __name__ == '__main__':
    # Configuration
    class Config:
        def __init__(self):
            self.seq_len = 10
            self.pred_len = 1
            self.channel = 5
            self.hidden_size = 64
            self.num_layers =2
            self.embedding_dim = 64
            self.patch_size = 5
            self.stride = 1
            self.conv1d_kernel_size = 3
            self.num_heads = 4
            self.num_blocks = 2

    config = Config()

    # Create model
    model = P_sLSTM(
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        channel=config.channel,
        hidden_size = config.hidden_size, 
        num_layers = config.num_layers, 
        embedding_dim=config.embedding_dim,
        patch_size=config.patch_size,
        stride=config.stride,
        conv1d_kernel_size=config.conv1d_kernel_size,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks
    )

    # Create input tensor
    x = torch.randn(32, config.seq_len, config.channel)  # [B, L, C]

    # Forward pass
    output = model(x)
    print(output.shape)  # Should be [32, config.pred_len, config.channel]