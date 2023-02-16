"""
最简单的基于字典的匹配分词

原理
    1、利用trie_dict 做最大匹配分词
    2、支持正向、反向的词匹配顺序


"""
from sgd_nlp.core.common.trie_tree import Trie
from sgd_nlp.core.common.read_file import yield_tokens


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


if __name__ == '__main__':
    r"""simple code test """
    import os
    data_home = r"C:\Users\SGD\Desktop\sgd-代码库\sgd_deep_learning_framwork\sgd_nlp\data"
    sub_dir = 'tokenizer'
    dict_name = 'test_default_dict'
    dpath = os.path.join(data_home, sub_dir, dict_name)

    tokenizer = MaxMatchTokenizer(dict_path=dpath, seq_reverse=False)
    tokenizer_reverse = MaxMatchTokenizer(dict_path=dpath, seq_reverse=True)

    seq1 = '我是周杰伦，唱一首七里香'
    seq2 = '工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作'

    def test_tokenizer(seq):
        res = tokenizer.forward(seq)
        reverse_res = tokenizer_reverse.forward(seq)

        print('*--*'*10)
        print('原句子:\t', seq)
        print('分词结果:\t', res)
        print('逆序分词结果:\t', reverse_res, end='\n'*2)

    test_tokenizer(seq1)
    test_tokenizer(seq2)
