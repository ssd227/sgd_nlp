from collections import Counter
from math import log

from .default_token import DefaultToken


class LanguageModel:
    """
        概率语言模型

        处理N阶语言模型的统计信息
        N-gram = 2，词对[x1,x2]的统计信息
        N-gram = 3，词对[x1,x2,x3]的统计信息
        ... ...

        todo 统计信息N向下兼容（todo设计思路）

        todo ngram的问题：
            n越大
                需要存的index就越多，内存吃紧
                但是统计数据越发稀疏，miss率高
    """

    def __init__(self, tokens_list, N_gram=0):
        self._eos = DefaultToken.eos
        self._bos = DefaultToken.bos
        self._padding = DefaultToken.padding

        self.minimum_prob = 1e-8

        self.N_gram = N_gram  # ngram的长度（标量）

        self.ngram_counter = [Counter() for _ in range(N_gram)]
        for tokens in tokens_list:
            self._build_ngram_counter(tokens)

    def _build_ngram_counter(self, tokens):
        padding_tokens = self.padding_tokens(tokens)
        Counter(padding_tokens)

        # add START STOP padding
        for gram_len_id in range(self.N_gram):
            """
            逻辑说明：
                ------- example code
                N = 3  # 输出grams的长度
                tokens = [0,1,2,3,4,5,6,7]
                end_idx = len(tokens)

                tuples = [tokens[i: end_idx-N+i+1] for i in range(N)]
                print(tuples)
                print([grams for grams in zip(*tuples)])
                ------  输出
                    [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
                    [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7)]
            """
            end_idx = len(padding_tokens)
            tuples = [padding_tokens[i: (end_idx - self.N_gram + i + 1)] for i in range(gram_len_id + 1)]
            self.ngram_counter[gram_len_id] += Counter([grams for grams in zip(*tuples)])

    def count(self, gram_seq):
        # input type preprocess
        if isinstance(gram_seq, list):
            gram_seq = tuple(gram_seq)
        if isinstance(gram_seq, str):
            gram_seq = (gram_seq,)

        gram_len = len(gram_seq)
        assert gram_len <= self.N_gram

        if gram_seq not in self.ngram_counter[gram_len - 1]:
            return 0  # 查询key不存在

        return self.ngram_counter[len(gram_seq) - 1][gram_seq]

    def count_all_token(self):
        return sum(self.ngram_counter[0].values())

    def prob(self, gram_seq):
        gram_len = len(gram_seq)
        assert gram_len <= self.N_gram

        if gram_len == 1:
            pred_c = self.count(gram_seq)
            candi_c = self.count_all_token()
            return pred_c / (candi_c + self.minimum_prob)  # add minimum_prob(1e-8) avoid dividing 0
        else:
            pred_c = self.count(gram_seq)
            candi_c = self.count(gram_seq[:-1])
            return pred_c / (candi_c + self.minimum_prob)

    def log_prob(self, gram_seq):
        return log(max(self.prob(gram_seq),
                       self.minimum_prob))  # for invalid input 0

    def padding_tokens(self, tokens):
        return [self._bos] * (self.N_gram - 1) + tokens + [self._eos]

    def reverse_padding_tokens(self, tokens):
        return tokens[self.N_gram - 1:-1]

    def padding_str(self, string):
        return self._bos * (self.N_gram - 1) + string + self._eos

