"""
LSTM

"""

import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
       
        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
            done by nn.Linear
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.W_ih = nn.Linear(in_features=input_size, out_features=4*hidden_size, bias=self.bias, device=device, dtype=dtype)
        self.W_hh = nn.Linear(in_features=hidden_size, out_features=4*hidden_size, bias=self.bias, device=device, dtype=dtype)


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        B, C_in = X.shape
        C_out = self.hidden_size
        
        if h is None:
            h = (torch.zeros(B, C_out, device=X.device, dtype=X.dtype),
                 torch.zeros(B, C_out, device=X.device, dtype=X.dtype),)
        h0, c0 = h
        
        xw = self.W_ih(X)
        hw = self.W_hh(h0)
        
        i,f,g,o = [x.reshape((B, C_out)) for x in torch.split((xw+hw).reshape((B,4,C_out)), split_size_or_sections=1, dim=1)]
        
        i= nn.Sigmoid()(i) # input gate
        f= nn.Sigmoid()(f) # forget gate
        o= nn.Sigmoid()(o) # output gate
        g = nn.Tanh()(g) # current generated value
        
        c_ = f*c0 + i*g
        h_ = o * nn.Tanh()(c_)
        
        return h_, c_


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype=None):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        
        self.lstm_cells = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.lstm_cells.append(LSTMCell(input_size, hidden_size,bias=bias,device=device,dtype=dtype))
            else:
                self.lstm_cells.append(LSTMCell(hidden_size, hidden_size,bias=bias,device=device,dtype=dtype))

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        T, B, C_in = X.shape # [Time_seq, Batch, C_in]
        C_out = self.hidden_size
        
        if h is None:
            h = (torch.zeros(self.num_layers, B, C_out, device=X.device, dtype=X.dtype),
                    torch.zeros(self.num_layers, B, C_out, device=X.device, dtype=X.dtype))
        h0, c0 = h
        hs = h0 # h[num_layers]
        cs = c0 # h[num_layers]
        
        output = [] # 时间维度，收集 h_highlayer_t， 共T个。
        for t in range(T):
            hs_tmp, cs_tmp = [], []    # 每个时间t下，收集一次 h_layer_i, 共num_layer个
            for ly in range(self.num_layers):
                if ly==0:
                    h, c = self.lstm_cells[ly](X = X[t].reshape((B,C_in)), 
                                              h = (hs[ly].reshape((B, C_out)),  cs[ly].reshape((B, C_out))))
                else:
                    h, c = self.lstm_cells[ly](X = hs_tmp[-1].reshape((B, C_out)),
                                            h = (hs[ly].reshape((B, C_out)),  cs[ly].reshape((B, C_out))))
                hs_tmp.append(h)
                cs_tmp.append(c)
                
            output.append(hs_tmp[-1]) # 收集顶层rnn_cell的输出h
            hs = torch.stack(hs_tmp, dim = 0) # use hidden state(at t-1) as input hs (at t)
            cs = torch.stack(cs_tmp, dim = 0)
        
        return torch.stack(output, dim=0), (hs, cs)