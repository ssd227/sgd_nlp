# encoding=utf8
import math
import random
from collections import Counter
from .default_token import DefaultToken

"""
# todo 仿照pytorch的Vocab 写个看看

padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 256
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

text_transform = T.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path),
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
)

"""


class Vocab:
    """ 提供词和词的简单映射

        功能：记录所有词和(词《=》id)映射关系, 映射(句子和ids),补充上default token
        具有动态的词统计信息(那就需要版本号了,不然旧的映射无法翻译) 是个问题,(到底需不需要这个功能呢?????)

        1、分析词频率
        2、提供index和word见的映射 map
        4、序列化state,不用重头处理
    """

    def __init__(self, tokens, min_freq=0, reserved_tokens=[]):
        self.unk_token = DefaultToken.unk_token  # todo 用这个做padding行不行
                
        # 统计词频率
        self._token_freqs = sorted(Counter(tokens).items(), key=lambda x: x[1], reverse=True)


        # 未知词元的索引为0
        self.idx_to_token = [self.unk_token] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        # 处理tokens
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

        # 统计总词数
        self.corpus_word_count = self._get_corpus_word_count()
        self._token_weight = [1 / self.corpus_word_count] + [freq / self.corpus_word_count for _, freq in
                                                             self._token_freqs]

        # 计算采样权重
        def adjust_probability(count):
            return 1-math.sqrt(1e-5/count) # 频数变换，改变采样概率
        new_token_freq = [1] + [freq for _, freq in self._token_freqs] # 补上默认unk字符
        self._token_weight_2 = list(map(adjust_probability, new_token_freq))
        
        self.sample_cache = [] 

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        return self.to_ids(tokens)

    # tokens转ids
    def to_ids(self, tokens):
        # 单个token直接查表
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.token_to_idx[self.unk_token])

        # 列表型数据,递归调用自己
        # ！还有递归效果,可以嵌套的转换index。但使用时需要用户注意,不然超出预期
        return [self.to_ids(token) for token in tokens]

    # ids转tokens
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    # 总token数
    def token_num(self):
        return len(self.idx_to_token)

    # 语料总词数
    def _get_corpus_word_count(self):
        return sum([x[1] for x in self._token_freqs])
    
    # 各token频率分布
    def token_freqs(self):
        return self._token_freqs

    def sample_word(self, k): # 按权重采样
        # 使用缓存稍微优化一下batch的吞吐量
        if len(self.sample_cache) < k:
            self.sample_cache += random.choices(self.idx_to_token, weights=self._token_weight_2, k=1000)
        
        # cache中还有余量
        ks = self.sample_cache[:k]
        self.sample_cache = self.sample_cache[k:]
        return ks

    def log_info(self):
        print('****** VOCAB LOG INFO ******')
        print('corpus_word_num: {}\nvocab_size: {}\nword_freq_count: \n{}...\n...{}'.format(
            self.corpus_word_count,
            len(self.idx_to_token),
            self._token_freqs[:50], self._token_freqs[-50:]))
        print('****** vocab log end ******')   

# factory function, return class Vocab
def vocab(tokens, min_freq: int = 1) -> Vocab:
    return Vocab(tokens, min_freq=min_freq)
