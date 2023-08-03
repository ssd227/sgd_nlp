"""
字典树

应用场景：
    基于字典的中文分词（用于存储字典中的词）

    todo 性能测试
"""

import collections


class TrieNode(object):
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False


class Trie(object):
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token):
        # print('insert token: ', token)
        cur_node = self.root
        for letter in token:
            cur_node = cur_node.children[letter]
        cur_node.is_word = True

    def insert_batch(self, tokens):
        # todo 优化插入效率
        for token in tokens:
            self.insert(token)

    # 字典中是否包含 待检索词
    def is_existed(self, word):
        cur_node = self.root
        for letter in word:
            cur_node = cur_node.children.get(letter)
            if cur_node is None:
                return False
        return cur_node.is_word

    def all_words_ids_pair(self, strings):
        """
        call by  生成 基于概率的-分词-paths

        input：
            指定字符串
        return：
            生成词的pairs（bid，eid）
                图邻接表（type-dict）
        """
        id_pairs = {}
        for b_id in range(len(strings)):
            id_pairs[b_id] = self._all_end_ids_from_head_is_word(strings, b_id)  # id_pairs[b_id] = [e_ids]
        return id_pairs

    # 字典中是否存在 以检索词开始的词
    def _all_end_ids_from_head_is_word(self, string, head_id):
        """
        return：
            从 head_id 开始的所有词对的 eid list
        """
        end_ids = []
        cid = head_id
        cur_node = self.root
        c = string[cid]

        while cur_node.children.get(c):
            cur_node = cur_node.children.get(c)
            if cur_node.is_word:
                end_ids.append(cid)

            cid += 1
            if cid >= len(string): break
            c = string[cid]

        # 单字 的词
        if end_ids == [] or head_id != end_ids[0]:
            end_ids = [head_id] + end_ids

        return end_ids

    def get_candidates_starts_with(self, input_query):
        """
        call by query_auto_fill

        宽度优先搜索
        感觉都挺费劲的，尤其是树的宽度过大
        能做但是效率太低了
        todo 待优化
            按照路径词总频率进行搜索剪枝
        """
        candi_query_list = []
        cur_node = self.root
        for letter in input_query:
            cur_node = cur_node.children.get(letter)
            if cur_node is None:
                return None

        # todo 不剪枝，不断复制prefix string，内存也要爆炸了
        # 宽度优先搜索队列 nodes_queue
        nodes_queue = [(cur_node.children, input_query)]  # (cur_node.children, prefix)
        while len(nodes_queue) > 0:
            tmp_nodes, prefix = nodes_queue.pop(0)
            for s, node in tmp_nodes.items():
                if node.is_word:
                    candi_query_list.append(prefix + s)
                nodes_queue.append((node.children, prefix + s))
        return candi_query_list

    def max_match(self, string):
        """
        call by最大匹配分词模型

        return 匹配的最长词段 split_id（划分index）
            比如：max_match_str = string[0:split_id]，不包含split_id

        备注：如果字典树没有对应词，中文直接划分出一个汉字作为独立词（最终split_id >= 1）
        """
        max_split_id = 1
        cur_node = self.root
        for i, c in enumerate(string):
            if c in cur_node.children:
                cur_node = cur_node.children.get(c)
                if cur_node.is_word:
                    max_split_id = i + 1
            else:
                break
        return max_split_id