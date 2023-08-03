"""
最简单的基于字典的匹配分词

原理
    1、利用trie_dict 做最大匹配分词
    2、支持正向、反向的词匹配顺序


"""
from sgd_nlp.common import Trie, yield_tokens

class MaxMatchTokenizer:
    def __init__(self, dict_path=None, seq_reverse=False):
        self.seq_reverse = seq_reverse  # （前序最大匹配vs逆序最大匹配）
        self.trie = Trie()

        # load default_dict file
        self.load_word_dict(
            yield_tokens(dict_path, reverse=seq_reverse))

    def load_word_dict(self, tokens):
        self.trie.insert_batch(tokens)

    def forward(self, string):
        if self.seq_reverse:
            string = string[::-1]
        tokens = []

        "=================- main process -================="
        left_str = string
        while len(left_str) > 0:
            split_id = self.trie.max_match(left_str)
            max_match_str, left_str = left_str[:split_id], left_str[split_id:]
            tokens.append(max_match_str)
        "=================- end main process -================="

        if self.seq_reverse:
            tokens = [token[::-1] for token in tokens[::-1]]
        return tokens

    def forward_batch(self, strings):
        # todo 多线程提高 Throughput
        pass



