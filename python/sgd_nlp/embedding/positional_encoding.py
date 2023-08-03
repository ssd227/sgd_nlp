"""
位置编码
模拟二进制
位置  0    1    2     3     4     5    6   7
编码  000  001  010   011   100   101  110 111


行代表词元在序列中的位置，列代表位置编码的不同维度
每个列维度使用不同的频率，变换频率依次降低。 类似于二进制的高阶数字变化较慢（100）
词元的行位置交替使用sin cos来生存每一个列维度对应的具体数值
"""
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, max_len=1000, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))  # batch_size, pos_num, emb_nums

        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        if num_hiddens // 2 == 0:
            self.P[:, :, 1::2] = torch.cos(X)
        else:
            self.P[:, :, 1::2] = torch.cos(X[:, :-1])

    def forward(self, X):
        X += self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
