"""
RNN

"""

import torch
from torch import nn

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
            done by nn.Linear
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        
        # 默认激活函数
        self.activate_f = nn.Tanh()
        if nonlinearity == "relu":
            self.activate_f = nn.ReLU()
        elif nonlinearity == "tanh":
            self.activate_f = nn.Tanh()
        
        self.W_ih = nn.Linear(in_features=input_size, out_features=hidden_size, bias=self.bias, device=device, dtype=dtype)
        self.W_hh = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=self.bias, device=device, dtype=dtype)

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        if h is None:
            h = torch.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
            
        return self.activate_f(self.W_ih(X)+self.W_hh(h))



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype=None):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn_cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype))
            else:
                self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))


    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        T, B, C_in = X.shape # [Time_seq, Batch, C_in]
        
        if h0 is None:
            h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=X.device, dtype=X.dtype)
        
        hs = h0
        # hs = ops.split(h0, axis=0) # h[num_layers]
        # Xs = ops.split(X, axis=0) # Xs[T]
        
        output = [] # 时间维度，收集 h_highlayer_t， 共T个。
        for t in range(T):
            hs_tmp = []     # 每个时间t下，收集一次 h_layer_i, 共num_layer个
            for ly in range(self.num_layers):
                if ly==0:
                    h = self.rnn_cells[ly](X[t].reshape((B,C_in)), hs[ly].reshape((B, self.hidden_size)))
                else:
                    h = self.rnn_cells[ly](hs_tmp[-1].reshape((B, self.hidden_size)), hs[ly].reshape((B, self.hidden_size)))
                hs_tmp.append(h)
                
            output.append(hs_tmp[-1]) # 收集顶层rnn_cell的输出h
            hs = torch.stack(hs_tmp, axis=0) # use hidden state(at t-1) as input hs (at t)
        
        return torch.stack(output, axis=0), hs

