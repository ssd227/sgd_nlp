"""
基于一阶语言模型的概率分词(最大化分词路径的概率)

method:
    1、基于trie tree构建待分词句子的所有分词路径(图结构)
    2、train:统计训练样本中词转移概率
    3、inference:返回最大概率的分词路径
"""
import collections
from sgd_nlp.common import Trie, LanguageModel
from sgd_nlp.common.read_file import *


class MaxProbabilityPathTokenizer:
    """
    思路:
        step1、根据一段文字,按照词典,构建所有可能的 分割路径
        step2、按照 维特比 算法找出一条概率最大的 分割方法

    example:
        mpp_tokenizer = MaxProbabilityPathTokenizer()
        mpp_tokenizer.train()
        mpp_tokenizer.inference(target_string)
    """

    def __init__(self):
        self.trie = Trie()  # 记录词表,方便构造segment-path
        self.language_model = None  # N阶语言模型类(ngram:用于统计tokens前后状态转移概率)

    def clear(self):
        self.trie = Trie()
        self.language_model = None

        # todo 待优化
        # self.language_model.clear()

    # 新增概率统计
    def train(self, tokens_list):
        # todo add new tokens to trie dict
        # todo 同时处理1D 和2D的情况

        tokens = transform_tokens_list_to_tokens(tokens_list)

        # init trie and language_model
        self.trie.insert_batch(tokens)
        self.language_model = LanguageModel(tokens_list, N_gram=2)

    def inference(self, string):
        padding_str = self.language_model.padding_str(string)
        if self.language_model is None:
            print("!!! language model is None !!!, please run train(tokens_list) first.")

        segment_paths = self._build_segment_paths(padding_str)
        padding_segment_result, max_prob = self._viterbi_alg(padding_str, segment_paths)
        segment_result = self.language_model.reverse_padding_tokens(padding_segment_result)
        return segment_result, max_prob

    def _build_segment_paths(self, string):
        # 构造一个句子的所有分词路径(有向无环图/DAG-邻接表)
        return self.trie.all_words_ids_pair(string)

    def _viterbi_alg(self, string, segment_paths):
        edges_bos, edges_eos = collections.defaultdict(list), collections.defaultdict(list)  # index
        for b_id, e_ids in segment_paths.items():
            for e_id in e_ids:
                edges_bos[b_id].append((b_id, e_id))
                edges_eos[e_id].append((b_id, e_id))
        edges_eos[-1] = [(-1, -1)]  # 遍历前序词时,给dp提供一个初始值dp[(0,0)] = 0

        # 动态规划
        dp, back_check_point = {}, {}  # dp state
        dp[(-1, -1)] = 0
        for idx in range(len(string)):
            for y1, y2 in edges_bos[idx]:
                w2 = string[y1: y2 + 1]

                max_log_prob = None

                for x1, x2 in edges_eos[y1 - 1]:
                    w1 = string[x1:x2 + 1]
                    log_prob = dp[(x1, x2)] + self.language_model.log_prob([w1, w2])

                    if max_log_prob is None:
                        max_log_prob = log_prob
                        back_check_point[(y1, y2)] = (x1, x2)
                    elif log_prob > max_log_prob:
                        max_log_prob = log_prob
                        back_check_point[(y1, y2)] = (x1, x2)

                dp[(y1, y2)] = max_log_prob

        # 反向过程:找概率最大的路径
        max_prob_end_point = max([(dp[y1y2], y1y2) for y1y2 in edges_eos[len(string) - 1]],
                                 key=lambda x: x[0])
        end_prob, end_ids = max_prob_end_point

        segment_result = []
        cur_node = end_ids
        while cur_node[0] >= 0:
            segment_result.append(cur_node)
            cur_node = back_check_point[cur_node]
        segment_result.reverse()

        return [string[id1:id2 + 1] for id1, id2 in segment_result], end_prob