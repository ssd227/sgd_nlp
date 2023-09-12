# encoding=utf8

"""
模型用法: apps/embedding
"""
import torch
from torch import nn
import torch.nn.functional as F

from torch import nn

class Glove(nn.Module):

    def __init__(self, token_num, emb_dim, sparse_emb=False):
        super(Glove, self).__init__()
        self.emb_i = nn.Embedding(token_num, embedding_dim=emb_dim, sparse=sparse_emb)
        self.emb_j = nn.Embedding(token_num, embedding_dim=emb_dim, sparse=sparse_emb)

        # 需要注册参数，才能在to device时一并转换
        self.bi = torch.nn.Parameter(torch.zeros((token_num, 1), dtype=torch.float64))
        self.bj = torch.nn.Parameter(torch.zeros((token_num, 1), dtype=torch.float64))

        self.x_max = 100
        self.alpha = 0.75  # 3/4 in paper

    def forward(self, x):
        """
        x:
        [idi idj xij]  wi_id wj_id 共现统计数
        维度：[B, 3]
        只要输入稀疏的词 共现对(Xij)的batch
        就可以直接训练
        """
        wi = self.emb_i(x[:, 0:1])  # [B, 1, emb_n]
        wj = self.emb_j(x[:, 1:2])  # [B, 1, emb_n]
        xij = x[:, 2]  # [B,1]

        wiwj = wi.bmm(wj.transpose(1, 2))  # [B, 1,1]
        log_xij = torch.log(xij)  # [B,1]
        bi = self.bi[x[:, 0:1]]  # [B, 1]
        bj = self.bj[x[:, 1:2]]  # [B, 1]

        fx = (xij / self.x_max) ** self.alpha  # [B, 1]
        fx[xij >= self.x_max] = 1  # 最大值截断

        mse = fx * torch.square(wiwj + bi + bj - log_xij)  # [B, 1]
        return mse
