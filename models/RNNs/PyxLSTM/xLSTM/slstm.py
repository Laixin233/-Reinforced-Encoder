"""
sLSTM: Scalar Long Short-Term Memory

This module implements the sLSTM (scalar LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The sLSTM extends the traditional LSTM by using exponential gating and a new memory mixing technique,
allowing for improved performance on various sequence modeling tasks.

Author: Mudit Bhargava
Date: June 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class sLSTM(nn.Module):
    """
    sLSTM layer implementation.

    This layer applies multiple sLSTM cells in sequence, with optional dropout between layers.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
        num_layers (int): Number of sLSTM layers.
        dropout (float, optional): Dropout probability between layers. Default: 0.0.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([sLSTMCell(input_size if i == 0 else hidden_size, hidden_size) 
                                     for i in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass of the sLSTM layer.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_state (tuple of Tensors, optional): Initial hidden state. Default: None.

        Returns:
            tuple: Output sequence and final hidden state.
        """
        batch_size, seq_length, _ = input_seq.size()
        
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)

        hidden_state = list(hidden_state)
        h_t, c_t = hidden_state[0], hidden_state[1]

        # h_t, c_t = [h.clone() for h in hidden_state[0]], [c.clone() for c in hidden_state[1]]
        
        outputs = []
        h_list = []
        c_list = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            h_layer = []
            c_layer = []
            for layer_idx, layer in enumerate(self.layers):
                h_l, c_l = h_t[layer_idx],c_t[layer_idx]
                h_l_new, c_l_new = layer(x, (h_l, c_l))
                h_layer.append(h_l_new)
                c_layer.append(c_l_new)
                h_l = h_l_new
                c_l = c_l_new

                x = self.dropout_layer(h_l) if layer_idx < self.num_layers - 1 else h_l

            h_t, c_t = h_layer, c_layer
            h_list.append(torch.stack(h_layer,dim=0))
            c_list.append(torch.stack(c_layer,dim=0))
            
            outputs.append(x)
        hidden = torch.stack(h_list, dim=0)
        context = torch.stack(c_list, dim=0)
        last_hidden_state = (hidden[-1], context[-1])    
        out = torch.stack(outputs, dim=0).permute(1, 0, 2)
        
        return out, last_hidden_state

    def init_hidden(self, batch_size):
        """Initialize hidden state for all layers."""
        return [(torch.zeros(batch_size, self.hidden_size, device=self.layers[0].weight_ih.device),
                 torch.zeros(batch_size, self.hidden_size, device=self.layers[0].weight_ih.device))
                for _ in range(self.num_layers)]

class sLSTMCell(nn.Module):
    """
    sLSTM cell implementation.

    This cell uses exponential gating as described in the xLSTM paper.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
    """

    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(4 * hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias)

    def forward(self, input, hx):
        """
        Forward pass of the sLSTM cell.

        Args:
            input (Tensor): Input tensor of shape (batch_size, input_size).
            hx (tuple of Tensors): Previous hidden state and cell state.

        Returns:
            tuple: New hidden state and cell state.
        """
        h, c = hx
        gates = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        
        i, f, g, o = gates.chunk(4, 1)
        
        i = torch.exp(i)  # Exponential input gate
        f = torch.exp(f)  # Exponential forget gate
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c = f * c + i * g
        h = o * torch.tanh(c)
        
        return h, c