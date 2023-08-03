import torch
import math
from torch import nn
from typing import Optional, Any, Union, Callable
from torch import Tensor
import torch.nn.functional as F

from .transformer import TransformerEncoder, PositionalEncoding


class Bert(nn.Module):
    r"""
    bert 模型实现
    """
    def __init__(self, vocab_size, 
                d_model: int = 768, nhead: int = 12, num_encoder_layers: int = 12,
                dim_feedforward: int = 2048, dropout: float = 0.1,
                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                device=None, dtype=None) -> None:
        
        factory_kwargs = {'d_model':d_model, 'nhead':nhead, 'dim_feedforward':dim_feedforward,
                          'dropout':dropout, 'activation': activation, 'layer_norm_eps':layer_norm_eps,
                          'batch_first':batch_first, 'norm_first':norm_first, 'device': device, 'dtype': dtype}
        super().__init__()   
        self.encoder = TransformerEncoder(num_layers=num_encoder_layers, **factory_kwargs)
        
        # todo：下面两个emb到底是不是可训练的参数（原始论文里也没详细说明）
        self.pos_emb = PositionalEncoding(d_model, dropout=dropout)
        self.seg_emb = SegmentEmbeddings(d_model, dropout=dropout)
        self.word_emb = nn.Embedding(vocab_size, d_model)
        
        
    def forward(self,
                first_sentence_ids,
                last_sentence_ids, 
                src_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        # word embedding
        xa, xb = [self.self.word_emb(ids)
                  for ids in [first_sentence_ids, last_sentence_ids]]  # 已经包含了特殊字符 [Batch,Time_series,Dim]
        # segment embedding
        xa =  self.seg_emb(xa, first_sentence=True)
        xb = self.seg_emb(xb, first_sentence=False)
        xs = torch.cat(xa, xb, dim=1) # 时间序列维度拼接
        # positional embedding
        src = self.pos_emb(xs)
        
        # encoder
        out = self.encoder(src, src_padding_mask)
        return out


class SegmentEmbeddings(nn.Module):
    r"""
    句子编码
    """
    def __init__(self, num_hiddens, max_len=1000, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.num_hiddens = num_hiddens
        self.dropout = nn.Dropout(dropout)
        self.seg_emb = nn.Parameter(torch.randn((2, 1, num_hiddens)))  # 2, pos_num, emb_nums
        
    def forward(self, X, first_sentence):
        id = 0 if first_sentence else 1
        X += self.seg_emb[id].reshape(1,self.num_hiddens).to(X.device)  # 会自动broadcast
        return self.dropout(X)

#####################################################################################
##################################  辅助函数   #######################################

# 300M参数的bert
def bert_large(vocab_size, device=None, dtype=None):
    return Bert(vocab_size, d_model=1024, nhead = 16, num_encoder_layers = 24,
                dim_feedforward= 2048, dropout= 0.1, device=device, dtype=dtype)

# 论文里用于和gpt比较的模型 100M参数
def bert_base(vocab_size, device=None, dtype=None):
    return Bert(vocab_size, d_model=768, nhead = 12, num_encoder_layers = 12,
                dim_feedforward= 2048, dropout= 0.1, device=device, dtype=dtype)

# 训练数据替换 todo 
#   15%抽调替换 [mask]80%, rand 10%, 原始token 10%

