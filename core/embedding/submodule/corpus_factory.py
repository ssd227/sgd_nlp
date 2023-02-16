# encoding=utf8
"""
对文本数据进行窗口采样

正样本： 来自词的上下文context
负样本： 随机从全部词样本中抽取

可以在均匀分布的前提下，按照词出现的概率采样
可以调节采样（抑制高词频，提高低词频）
"""
import collections
import random
import re
import torch
import os

from sgd_nlp.core.common.vocab import Vocab
from sgd_nlp.core.common.read_file import yield_tokens_from_docs, yield_lines, yield_tokens_from_line


# todo 需不需要拆分成三个子类使用继承关系

class CorpusFactory:
    def __init__(self, document_dir_path):
        self.docs_dir = document_dir_path
        self.documents: list = self._init_documents(document_dir_path)
        self.vocab = self._init_vocab(self.documents)

    def _init_documents(self, docs_dir):
        """递归枚举出所有子文件路径"""

        def recursive_list_files(root_dir):
            f_list = []
            for root, dirs, files in os.walk(root_dir, topdown=False):
                f_list += ([os.path.join(root, fi) for fi in files])
                if dirs:  # 非空
                    for di in dirs:
                        f_list += (recursive_list_files(os.path.join(root, di)))
            return f_list

        if os.path.isfile(docs_dir):  # case1： 单个文件
            return [docs_dir]
        return recursive_list_files(docs_dir)  # case2：递归遍历文件夹

    def _init_vocab(self, docs):
        tokens = yield_tokens_from_docs(docs, preprocess_func=replace_rule)
        return Vocab(tokens)

    def token_num(self):
        return self.vocab.token_num()

    def iter_all_corpus_lines(self):
        for doc in self.documents:
            for line in yield_lines(doc):
                yield line  # 返回文件row line

    def random_one_line(self):
        return self.random_line()[0]

    def random_line(self, keep_num=1) -> list:
        # 随机采样返回一行文本raw (todo 不能保证各文件的均匀采样)
        random_file = random.choice(self.documents)

        # 数据流采样（水塘采样）
        random_line = reservoir_sampling(data_stream=yield_lines(random_file), keep_num=keep_num)
        return random_line

    def neg_sample(self, k):
        # rule1：按照vocab里记录的文本词的概率，按均匀抽样的方式返回k个值
        return self.vocab.sample_word(k)

        # todo 1、压缩大概率的词 比如the会出现非常多
        # todo 2、使用log压缩采样的级数


class CorpusFactoryGlove(CorpusFactory):
    def __init__(self, document_dir_path, win_width, pair_symmetric):
        super(CorpusFactoryGlove, self).__init__(document_dir_path)

        # set variable and flag
        self.win_width = win_width
        self.pair_symmetric = pair_symmetric
        self.cur_index = 0

        # some calc
        self.xij = self._init_xij()  # 全局共现词对的统计表-稀疏矩阵

    def window_sample(self, tokens: list):
        # 返回tokens中，中心词和周围词的词对
        res = []
        half_win = self.win_width // 2

        if self.pair_symmetric:
            # CASE1- 词对对称的采样:
            #   当前词与左右两个 half_win的关系
            #       有正向词对 也有 反向词对
            for ci in range(len(tokens)):
                context_w = tokens[max(ci - half_win, 0):ci] + tokens[ci + 1:ci + half_win + 1]
                w_pairs = [(tokens[ci], cont_w) for cont_w in context_w]
                res += w_pairs
            return res

        # CASE2- 词非对称的采样:
        #   暂时定义为当前词与右边win_width的单项pair关系
        for ci in range(len(tokens)):
            right_context_w = tokens[ci + 1: ci + self.win_width + 1]
            w_pairs = [(tokens[ci], cont_w) for cont_w in right_context_w]
            res += w_pairs
        return res

    def _init_xij(self):
        xij = collections.defaultdict(lambda: 0)  # 默认值为0的统计dict

        for raw_line in self.iter_all_corpus_lines():
            tokens = [replace_rule(token) for token in raw_line.strip().split()]  # 清洗标点符号的特殊字符
            word_pairs = self.window_sample(tokens)
            for wp in word_pairs:
                xij[wp] += 1

        xij = sorted(xij.items(), key=lambda x: x[1], reverse=True)
        return xij

    def training_batch(self, batch_num, device):
        # 随机抽batch
        b_x = random.choices(self.xij, k=batch_num)  # [(wi,wj), xij_count]

        # 2 batch截断 and id替换
        b_id = [self.vocab[wp[0]] + [wp[-1]] for wp in b_x]

        # 3 transform to pytorch device
        return torch.tensor(b_id, dtype=torch.int64).to(device)

    def word_pairs_num(self):
        return len(self.xij)


class CorpusFactorySkipGram(CorpusFactory):
    def __init__(self, document_dir_path):
        super(CorpusFactorySkipGram, self).__init__(document_dir_path)

    def window_sample(self, tokens: list, win_width=5):
        # 返回tokens中，中心词和周围词的词对
        res = []
        half_win = win_width // 2

        for ci in range(len(tokens)):
            cur_context_w = tokens[max(ci - half_win, 0):ci] + tokens[ci + 1:ci + half_win + 1]
            cur_w_pairs = [[tokens[ci], cont_w] for cont_w in cur_context_w]
            res += cur_w_pairs
        return res

    def training_batch(self, batch_num, device, win_width, neg_k):
        b_x = []

        # 1 sample line one by one
        while len(b_x) < batch_num:
            tmp_line = self.random_line()[0]
            tmp_tokens = yield_tokens_from_line(tmp_line, preprocess_func=replace_rule)
            tmp_x = self.window_sample(tmp_tokens, win_width=win_width)

            for x in tmp_x:
                x += self.neg_sample(k=neg_k)
            b_x += tmp_x

        # 2 batch截断 and id替换
        b_id = [self.vocab[tokens] for tokens in b_x[:batch_num]]

        # 3 transform to pytorch device
        return torch.tensor(b_id, dtype=torch.int64).to(device)


class CorpusFactoryCbow(CorpusFactory):
    def __init__(self, document_dir_path):
        super(CorpusFactoryCbow, self).__init__(document_dir_path)

    def window_sample(self, tokens: list, win_width):
        # 返回tokens中，中心词和周围词的词
        res = []
        half_win = win_width // 2

        for ci in range(len(tokens)):
            cur_w = [tokens[ci]] + tokens[max(ci - half_win, 0):ci] + tokens[ci + 1:ci + half_win + 1]
            # padding
            if len(cur_w) != win_width:
                cur_w += [self.vocab.unk_token for _ in range(win_width - len(cur_w))]

            res.append(cur_w)
        return res

    def training_batch(self, batch_num, device, win_width, neg_k):
        b_x = []

        # 1 sample line one by one
        while len(b_x) < batch_num:
            tmp_line = self.random_line()[0]
            tmp_tokens = yield_tokens_from_line(tmp_line, preprocess_func=replace_rule)
            tmp_x = self.window_sample(tmp_tokens, win_width=win_width)  # different from skip gram

            for x in tmp_x:
                x += self.neg_sample(k=neg_k)
            b_x += tmp_x

        # 2 batch截断 and id替换
        b_id = [self.vocab[tokens] for tokens in b_x[:batch_num]]

        # 3 transform to pytorch device
        return torch.tensor(b_id, dtype=torch.int64).to(device)


# token预处理函数 todo 预处理规则需要在类内部统一，待处理
def replace_rule(token):
    return re.sub(r"[.?,!()\}\{\[\]\"]", r'', token).lower()


def reservoir_sampling(data_stream, keep_num=1):
    res = []
    i = 0
    for item in data_stream:
        i += 1  # 计数
        # 以概率1保留前N各数
        if len(res) < keep_num:
            res.append(item)
        else:
            # 对于第i个数，以概率k/i保留， 并随机占据任意位置
            p_i = 1.0 * keep_num / i
            if random.random() <= p_i:  # [0, 1]随机分布
                # 随机替换前k个数中的一个
                rand_id = random.randint(0, keep_num - 1)
                res[rand_id] = item
    return res


if __name__ == '__main__':
    data_home = r'C:\Users\SGD\Desktop\sgd-代码库\sgd_deep_learning_framwork\sgd_nlp\data'
    sub_dir = r'friends\season01'

    corpus_dir = os.path.join(data_home, sub_dir)
    print('CURRENT PATH:\t', corpus_dir)
    win_width = 11
    pair_symmetric = False
    batch_num = 1024

    corpus_factory = CorpusFactoryGlove(corpus_dir,
                                        win_width=win_width,
                                        pair_symmetric=pair_symmetric)

    device = torch.device('cuda')
    corpus_factory.training_batch(batch_num=batch_num,
                                  device=device)
