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
from ..common.vocab import Vocab
from ..common.read_file import yield_tokens_from_docs, yield_lines, yield_tokens_from_line


# 基类
class CorpusFactory:
    def __init__(self, document_home_path):
        self.documents: list = self._init_documents(document_home_path) # 语料文件List
        self.vocab = self._init_vocab(self.documents) # 词表（字典）
        
        self.sample_lines_cache = [] # 提高采样速度，避免反复读文件只为了一行数据(低效)

    def _init_documents(self, docs_dir):
        # 递归枚举所有子文件路径
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
        else: # case2：递归遍历文件夹
            return recursive_list_files(docs_dir)  

    # 使用所有语料数据初始化 Vocab词表
    def _init_vocab(self, docs):
        tokens = yield_tokens_from_docs(docs, preprocess_func=replace_rule)
        return Vocab(tokens)

    # Vocab词表 单词数
    def token_num(self):
        return self.vocab.token_num()

    def random_one_line(self):
        return self.random_line(1)[0]

    # 使用cache优化
    def random_line(self, n) -> list:
        if len(self.sample_lines_cache) < n: # 缓存不足,补充样本lines
            random_file = random.choice(self.documents)
            # 数据流采样（水塘采样）
            self.sample_lines_cache += reservoir_sampling(lines_stream=yield_lines(random_file), max_keep_num=500)
        
        sample_lines= self.sample_lines_cache[:n]
        self.sample_lines_cache = self.sample_lines_cache[n:]
        return sample_lines

    def neg_sample(self, k):
        # rule1：按照vocab里记录的文本词的概率，按均匀抽样的方式返回k个值
        return self.vocab.sample_word(k)

        # todo 1、压缩大概率的词 比如the会出现非常多
        # todo 2、使用log压缩采样的级数
    
    '''------------------ help func -----------------------'''    
    def iter_all_corpus_lines(self):
        for doc in self.documents:
            for line in yield_lines(doc):
                yield line  # 返回文件row line

# glove派生类
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

        # 2 id替换
        b_id = [self.vocab[wp[0]] + [wp[-1]] for wp in b_x] # x: [idi idj xij]  wi_id wj_id 共现统计数 维度：[B, 3]

        # 3 transform to pytorch device
        return torch.tensor(b_id, dtype=torch.int64).to(device)

    def word_pairs_num(self):
        return len(self.xij)

# skip gram派生类
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

    def training_batch(self, batch_num, win_width, neg_k, device):
        b_x = []

        # 1 sample line one by one
        while len(b_x) < batch_num:
            tmp_line = self.random_one_line()
            tmp_tokens = yield_tokens_from_line(tmp_line, preprocess_func=replace_rule)
            tmp_x = self.window_sample(tmp_tokens, win_width=win_width)

            for x in tmp_x:
                x += self.neg_sample(k=neg_k)
            b_x += tmp_x

        # 2 batch截断、id替换
        b_id = [self.vocab[tokens] for tokens in b_x[:batch_num]]

        # 3 transform to pytorch device
        return torch.tensor(b_id, dtype=torch.int64).to(device)


# cbow派生类
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

    def training_batch(self, batch_num, win_width, neg_k, device):
        b_x = []

        # 1 sample line one by one
        while len(b_x) < batch_num:
            tmp_line = self.random_one_line()
            tmp_tokens = yield_tokens_from_line(tmp_line, preprocess_func=replace_rule)
            tmp_x = self.window_sample(tmp_tokens, win_width=win_width)  # different from skip gram

            for x in tmp_x:
                x += self.neg_sample(k=neg_k)
            b_x += tmp_x

        # 2 batch截断、id替换
        b_id = [self.vocab[tokens] for tokens in b_x[:batch_num]]

        # 3 transform to pytorch device
        return torch.tensor(b_id, dtype=torch.int64).to(device)


##################################################################################################
# token预处理函数 todo 预处理规则需要在类内部统一，待处理
def replace_rule(token):
    return re.sub(r"[.?,!()\}\{\[\]\"]", r'', token).lower()


def reservoir_sampling(lines_stream, max_keep_num=100):
    res = []
    i = 0
    for line in lines_stream:
        i += 1  # 计数
        # 以概率1保留前N各数
        if len(res) < max_keep_num:
            res.append(line)
        else:
            # 对于第i个数，以概率k/i保留， 并随机占据任意位置
            p_i = 1.0 * max_keep_num / i
            if random.random() <= p_i:  # [0, 1]随机分布
                # 随机替换前k个数中的一个
                rand_id = random.randint(0, max_keep_num - 1)
                res[rand_id] = line
    return res