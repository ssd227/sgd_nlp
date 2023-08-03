import torch
from torch import nn
from .rnn.gru import GRU
from .rnn.lstm import LSTM
from .rnn.rnn import RNN


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='lstm', device=None, dtype=torch.float32):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        self.hidden_size = hidden_size
        
        self.emb = nn.Embedding(num_embeddings=output_size, embedding_dim=embedding_size, device=device, dtype=dtype)
        
        if seq_model == "rnn": 
            self.seq_model = RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == "lstm":
            self.seq_model = LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == "gru":
            self.seq_model = GRU(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        else:
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        T,B = x.shape
        out, h = self.seq_model( self.emb(x) )
        y =  self.linear(out.reshape((T*B, self.hidden_size)))
        return y, h