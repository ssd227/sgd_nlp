from collections import Counter
from math import log

from sgd_nlp.core.common.default_token import DefaultToken


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


if __name__ == '__main__':
    " simple test code"


    def test_language_model(tokens_list):
        lan_model = LanguageModel(tokens_list=tokens_list, N_gram=3)

        flag_test_all = False

        flag_test_build_ngram_counters = False
        if flag_test_build_ngram_counters or flag_test_all:
            print('\n********** build ngram counters **********')
            for i in range(3):
                print(lan_model.ngram_counter[i])

        flag_test_tuple = False
        if flag_test_tuple or flag_test_all:
            # python 稀奇的小问题
            print('\n********** python 稀奇的小问题 **********',
                  "\ncase1:\t", (DefaultToken.unk_token), ' type is ', type((DefaultToken.unk_token)),
                  "\ncase2:\t", (DefaultToken.unk_token,), ' type is ', type((DefaultToken.unk_token,)), )

        flag_test_count_all_token = False
        if flag_test_count_all_token or flag_test_all:
            print("\n********** test: count_all_token() **********",
                  "\nall token num:\t", lan_model.count_all_token(),
                  "\ntoken dict:\t", lan_model.ngram_counter[0])

        flag_test_count = False
        if flag_test_count or flag_test_all:
            print("\n********** test: count()_1 **********",
                  "\ncount bos_token:\t", lan_model.count(DefaultToken.bos), )
            assert lan_model.count(DefaultToken.bos) == 8

            print("\n********** test: count()_2 **********",
                  "\ncount (a, b, c):\t", lan_model.count(['a', 'b', 'c']), )
            assert lan_model.count(['a', 'b', 'c']) == 1

        flag_test_prob = False
        if flag_test_prob or flag_test_all:
            print("\n********** test: prob()_1 **********",
                  "\nprob c|(a, b):\t", lan_model.prob(['a', 'b', 'c']), )
            assert abs(lan_model.prob(['a', 'b', 'c']) - 0.5) < 1e-5  # 2case (a b c) & (a b <-eos->)

            print("\n********** test: prob()_2 **********",
                  "\nprob f|e:\t", lan_model.prob(['e', 'f']), )
            assert abs(lan_model.prob(['e', 'f']) - 0) < 1e-5

            print("\n********** test: prob()_3 **********",
                  "\nprob b:\t", lan_model.prob('b'),
                  "\ncount b:\t", lan_model.count('b'),
                  "\ncount all token:\t", lan_model.count_all_token(),
                  "\nvalid prob:\t", lan_model.count('b') / lan_model.count_all_token(), )
            assert abs(lan_model.prob('b') - 0.142857142) < 1e-5

        flag_test_log_prob = True
        if flag_test_log_prob or flag_test_all:
            print("\n********** test: log_prob()_1 **********",
                  "\nlog_prob c|(a, b):\t", lan_model.log_prob(['a', 'b', 'c']),
                  "\nvalid log prob:\t", log(0.5), )
            assert abs(lan_model.log_prob(['a', 'b', 'c']) - -0.693147185) < 1e-5  # 2case (a b c) & (a b <-eos->)


    " simple test code main loop"
    doc = "a b c d\n c b a \n a b\n a"
    tokens_list = [line.strip().split() for line in doc.split('\n')]
    tokens = [token for tokens in tokens_list for token in tokens]

    print(tokens_list)
    print(tokens)

    test_language_model(tokens_list)
