from max_match_tokenizer import MaxMatchTokenizer
from statistics import mean


class BiMaxMatchTokenizer:
    def __init__(self, dict_path=None):
        self.tokenizer = MaxMatchTokenizer(dict_path=dict_path, seq_reverse=False)
        self.tokenizer_reverse = MaxMatchTokenizer(dict_path=dict_path, seq_reverse=True)

    def forward(self, strings):
        # 获取前序、后续分词结果(tokens)
        tokens_left_order = self.tokenizer.forward(strings)
        tokens_right_order = self.tokenizer_reverse.forward(strings)

        tokens = self._merge_with_rules(tokens_left_order, tokens_right_order)
        return tokens

    def _merge_with_rules(self, tokens1, tokens2):
        r"""
        定制化tokens的比较规则
        return：其中最合理的分词结果
        """
        # RULE1: 相同的分词长度，返回前序最大匹配分词结果
        if self._is_equal_obj(tokens1, tokens2): return tokens1

        # RULE2: 返回平均分词长度较大的tokens
        avg_len_1 = self._avg_list_len(tokens1)
        avg_len_2 = self._avg_list_len(tokens2)
        if avg_len_1 > avg_len_2: return tokens1
        if avg_len_1 < avg_len_2: return tokens2

        # 规则无法处理的其他情况，默认返回tokens1
        return tokens1

    def _is_equal_obj(self, li_a, li_b):
        if len(li_a) != len(li_b): return False
        for item_a, item_b in zip(li_a, li_b):
            if item_a != item_b: return False
        return True

    def _avg_list_len(self, li):
        return mean([len(x) for x in li])


if __name__ == '__main__':
    """ simple code test"""
    import os

    data_home = r"C:\Users\SGD\Desktop\sgd-代码库\sgd_deep_learning_framwork\sgd_nlp\data"
    sub_dir = 'tokenizer'
    dict_name = 'test_default_dict'
    dpath = os.path.join(data_home, sub_dir, dict_name)

    tokenizer = BiMaxMatchTokenizer(dpath)

    seq1 = '我是周杰伦，唱一首七里香'
    seq2 = '工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作'


    def test_tokenizer(seq):
        res = tokenizer.forward(seq)
        print('*--*' * 10)
        print('原句子:\t', seq)
        print('分词结果:\t', res, end='\n' * 2)


    test_tokenizer(seq1)
    test_tokenizer(seq2)
