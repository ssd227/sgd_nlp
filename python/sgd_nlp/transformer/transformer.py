import torch
import math
from torch import nn
from typing import Optional, Any, Union, Callable
from torch import Tensor
import torch.nn.functional as F


class Transformer(nn.Module):
    r"""
    
    todo:
    内部矩阵使用BTC [Batch, TimeSeris, Channel_dim]
    通过开关在底层模块实现
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                # custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                device=None, dtype=None) -> None:
        
        factory_kwargs = {'d_model':d_model, 'nhead':nhead, 'dim_feedforward':dim_feedforward,
                          'dropout':dropout, 'activation': activation, 'layer_norm_eps':layer_norm_eps,
                          'batch_first':batch_first, 'norm_first':norm_first, 'device': device, 'dtype': dtype}
        super().__init__()   
        self.encoder = TransformerEncoder(num_layers=num_encoder_layers, **factory_kwargs)
        self.decoder = TransformerDecoder(num_layers=num_decoder_layers, **factory_kwargs)
        self.pos_emb = PositionalEncoding(d_model, dropout=dropout)
        
        
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        # positional encoding
        src = self.pos_emb(src)
        tgt = self.pos_emb(tgt)
        
        B1,T1,C1 = src.shape
        B2,T2,C2 = tgt.shape
        
        if tgt_mask == None:
            tgt_mask = torch.triu(-float("inf")*torch.ones(T2, T2), 1)
        
        memory = self.encoder(src, src_mask)
        decoder_h = self.decoder(tgt, memory ,memory_mask)
        
        return decoder_h
        
    
#######################################################################################
################################  Encoder #############################################
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers:int, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        
        factory_kwargs = {'d_model':d_model, 'nhead':nhead, 'dim_feedforward':dim_feedforward,
                          'dropout':dropout, 'activation': activation, 'layer_norm_eps':layer_norm_eps,
                          'batch_first':batch_first, 'norm_first':norm_first, 'device': device, 'dtype': dtype}
        super().__init__()
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(**factory_kwargs) for _ in range(num_layers)])
    
    def forward(self, x, attn_mask=None):
        # attns = [] # self attention
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, attn_mask)
            # 疑问，这里不需要detach(), 计算图还可以从decoder返回到encoder?? 需要这么处理吗    
        return x
        

class TransformerEncoderLayer(nn.Module):
    r'''
    
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
    '''
    def __init__(self,  d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
    
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = activation

    def forward(self, x, attn_mask=None):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x,x,x, attn_mask=attn_mask)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

#######################################################################################
################################  Decoder #############################################

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers:int, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        
        factory_kwargs = {'d_model':d_model, 'nhead':nhead, 'dim_feedforward':dim_feedforward,
                          'dropout':dropout, 'activation': activation, 'layer_norm_eps':layer_norm_eps,
                          'batch_first':batch_first, 'norm_first':norm_first, 'device': device, 'dtype': dtype}
        super().__init__()
        self.num_layers = num_layers
        self.decoder_layers = [TransformerDecoderLayer(**factory_kwargs) for _ in range(num_layers)]
    
    def forward(self, x, h, attn_mask):
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, h, attn_mask)
        return x
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self,  d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                device=None, dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
    
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = activation

    def forward(self, x, h, attn_mask):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask)
            x = x + self._ca_block(self.norm2(x), h, attn_mask) 
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask))
            x = self.norm2(x + self._ca_block(x, h, attn_mask))
            x = self.norm3(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(K=x, Q=x, V=x, attn_mask=attn_mask)[0]
        return self.dropout1(x)
    
    # cross_attention block
    def _ca_block(self, x: Tensor,
                  h: Tensor,
                  attn_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(K=x, Q=h, V=h, attn_mask=attn_mask)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

#######################################################################################
################################  Attention ###########################################

class MultiheadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.0, bias=True, batch_first=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        
        self.dropout = nn.Dropout(dropout)
        
        self.proj_k = nn.Linear(emb_dim, emb_dim, bias=bias, **factory_kwargs)
        self.proj_q = nn.Linear(emb_dim, emb_dim, bias=bias, **factory_kwargs)
        self.proj_v = nn.Linear(emb_dim, emb_dim, bias=bias, **factory_kwargs)
        
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=bias, **factory_kwargs)
        
    def forward(self, K, Q, V, attn_mask=None, average_attn_weights=True):
        _,T1,_ = K.shape
        N,T,d = V.shape
        
        
        assert self.emb_dim == d , "self.emb_dim:[{}], d:[{}]".format(self.emb_dim, d)
        
        K = self.dropout(self.proj_k(K))
        Q = self.dropout(self.proj_q(Q))
        V = self.dropout(self.proj_v(V))
        
        K = K.reshape(N,T1,self.num_heads, self.emb_dim//self.num_heads).swapaxes(1,2) #[N, nums_heads, T1, d//num_heads]
        Q,V = [a.reshape(N,T,self.num_heads, self.emb_dim//self.num_heads).swapaxes(1,2) for a in (Q,V)]  #[N, num_heads, T, d//num_heads]
        
        attn = nn.Softmax(dim=-1)(K@Q.swapaxes(-1, -2) / math.sqrt(d//self.num_heads) + (attn_mask if attn_mask!=None else 0)) # 负无穷来mask不需要softmax的值
        
        return self.out_proj((attn@V).swapaxes(1,2).reshape(N,T1,d)), attn

# todo 只能计算self attention，不方便计算cross attention
# class MultiheadAttention(nn.Module):
#     def __init__(self, emb_dim, num_heads, dropout=0.0, bias=True, batch_first=False, device=None, dtype=None):
#         super().__init__()
        
#         assert emb_dim % num_heads == 0
#         self.emb_dim = emb_dim
#         self.num_heads = num_heads
        
#         self.dropout = dropout # todo 未使用
        
#         self.proj_kqv = nn.Linear(emb_dim, 3*emb_dim, bias=bias, device=device, dtype=dtype)
#         self.out_proj = nn.Linear(emb_dim, emb_dim, bias=bias, device=device, dtype=dtype)
        
#     def forward(self, X, attn_mask=None, average_attn_weights=True):
#         N,T,d = X.shape
#         assert self.emb_dim == d
        
#         K,Q,V = torch.split(self.proj_kqv(X), split_size_or_sections=d, dim=-1)
#         K,Q,V = [a.reshape(N,T,self.num_heads, self.emb_dim//self.num_heads).swapaxes(1,2) for a in (K,Q,V)]  #[N, num_heads, T, d//num_heads]
        
#         attn = nn.Softmax(dim=-1)(K@Q.swapaxes(-1, -2) / math.sqrt(d//self.num_heads) + (attn_mask if attn_mask!=None else 0)) # 负无穷来mask不需要softmax的值
        
#         return self.out_proj((attn@V).swapaxes(1,2).reshape(N,T,d)), attn
    
    
#######################################################################################
##############################  Positional Encoding  ##################################

class PositionalEncoding(nn.Module):
    r"""
    位置编码
    模拟二进制
    位置  0    1    2     3     4     5    6   7
    编码  000  001  010   011   100   101  110 111

    行代表词元在序列中的位置，列代表位置编码的不同维度
    每个列维度使用不同的频率，变换频率依次降低。 类似于二进制的高阶数字变化较慢（100）
    词元的行位置交替使用sin cos来生存每一个列维度对应的具体数值
    """
    
    def __init__(self, num_hiddens, max_len=1000, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))  # batch_size, pos_num, emb_nums

        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        
        self.P[:, :, 0::2] = torch.sin(X)
        if num_hiddens % 2 == 0:
            self.P[:, :, 1::2] = torch.cos(X)
        else:
            self.P[:, :, 1::2] = torch.cos(X[:, :-1])

    def forward(self, X):
        X += self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

