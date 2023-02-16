# encoding=utf8

"""
define two embedding models
    -Cbow
    -SkipGram

模型用法：application/embedding
"""
import time

import torch
from torch import nn
import torch.nn.functional as F

# todo cbow的训练难度更大一点，结果不容易收敛
class Cbow(nn.Module):
    def __init__(self, token_num, win_width, emb_dim=256, sparse_emb=False):
        super(Cbow, self).__init__()
        self.emb_i = nn.Embedding(token_num + 10, embedding_dim=emb_dim, sparse=sparse_emb)  # todo 为什么需要+10
        self.emb_o = nn.Embedding(token_num + 10, embedding_dim=emb_dim, sparse=sparse_emb)

        self.win_width = win_width

    def forward(self, x):
        """
        x: [xi, xo1,..., xon, , xn1,...,xn3]  B X window_size
        """
        center_x = self.emb_i(x[:, 0:1])  # [B, 1, emb_n]
        context_x = self.emb_o(x[:, 1:self.win_width])  # [B, win_width-1, emb_n]
        # neg_x = self.emb_o(x[:, self.win_width:])  # [B, neg_k, emb_n] #todo 注意 loss会差很多
        neg_x = self.emb_i(x[:, self.win_width:])  # [B, neg_k, emb_n]

        context_x = torch.sum(context_x, dim=1, keepdim=True)  # [B, 1, emb_n]
        center_x = center_x.transpose(1, 2)  # [B, emb_n, 1]
        neg_x = neg_x.transpose(1, 2)  # [B, emb_n, h]

        y_oi = context_x.bmm(center_x)  # [B, 1, 1]
        y_on = context_x.bmm(neg_x) * -1  # [B, 1, neg_k]
        y = torch.cat([y_oi, y_on], dim=2)  # [B, 1, 1+neg_k]

        y = F.logsigmoid(y)  # [B, 1, 1+neg_k]
        return y


class SkipGram(nn.Module):
    def __init__(self, token_num, emb_dim=256, sparse_emb=False):
        super(SkipGram, self).__init__()
        self.emb_i = nn.Embedding(token_num + 10, embedding_dim=emb_dim, sparse=sparse_emb)
        self.emb_o = nn.Embedding(token_num + 10, embedding_dim=emb_dim, sparse=sparse_emb)

    def forward(self, x):
        """
        x: [xi, xo, xn1,...,xn3]  shape:[B, 1+1+neg_k]
        """
        center_x = self.emb_i(x[:, 0:1])  # [B, 1, emb_n]
        context_x = self.emb_o(x[:, 1:])  # [B, 1+neg_k, emb_n]

        context_x = context_x.transpose(1, 2)  # [B, emb_n, 1+neg_k]

        y = center_x.bmm(context_x) * -1  # 负样本添加负号 [B, 1, 1+neg_k]
        y[:, :, :0] *= -1  # 正样本 设置为1

        y = F.logsigmoid(y)  # [B, 1, 1+neg_k]
        return y
