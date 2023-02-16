"""
位置编码
模拟二进制
位置  0    1    2     3     4     5    6   7
编码  000  001  010   011   100   101  110 111


行代表词元在序列中的位置，列代表位置编码的不同维度
每个列维度使用不同的频率，变换频率依次降低。 类似于二进制的高阶数字变化较慢（100）
词元的行位置交替使用sin cos来生存每一个列维度对应的具体数值
"""
import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    batchs, num_steps, encoding_dim = 1, 200, 31
    pos_encoding = PositionalEncoding(num_hiddens=encoding_dim, max_len=num_steps)
    pos_encoding.eval()

    X = torch.zeros((batchs, num_steps, encoding_dim))
    xe = pos_encoding(X)
    print(xe.shape)

    xx = torch.arange(num_steps)
    yy1 = xe[0, :, 4].reshape(-1)
    yy2 = xe[0, :, 10].reshape(-1)
    yy3 = xe[0, :, 12].reshape(-1)
    yy4 = xe[0, :, 20].reshape(-1)

    plt.plot(xx, yy1)
    plt.plot(xx, yy2)
    plt.plot(xx, yy3)
    plt.plot(xx, yy4)
    plt.show()
