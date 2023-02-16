"""
query自动补全

todo
    1、基于拼音
    2、极限压力测试 + 不同数据结构的性能分析。
    3、好像倒排索引也是可以做的，只要索引建立的够好，就没多少latency。
"""

from sgd_nlp.core.common.read_file import yield_lines
from sgd_nlp.core.common.trie_tree import Trie

"""
QueryAutoFill（先实现个简单版本）

输入：
    用户键盘输入的query(不要求完整)
功能：
    1、补全所有相关搜索（start with ***），列出所有候选的query
    2、不同的语料库可以生成不同的候选队列
        用户名、产品名、热门搜索词...

corner case：
    允许query带空格(预料中的一行作为一个query)
todo:
    当返回的结果太多时，按照热度对候选补全query做截断。（自行定义热度）  
"""


class QueryAutoFill:
    def __init__(self, candi_query_source_file):
        self.trie = Trie()

        # load default_query_source file
        self.load_candi_query_lines(
            yield_lines(candi_query_source_file))

    def load_candi_query_lines(self, query_lines):
        self.trie.insert_batch(query_lines)

    def forward(self, input_query):
        new_query = input_query.strip()
        if len(new_query) == 0:
            return None
        return self.trie.get_candidates_starts_with(new_query)


if __name__ == '__main__':
    r"""simple code test"""
    import os

    data_home = r'C:\Users\SGD\Desktop\sgd-代码库\sgd_deep_learning_framwork\sgd_nlp\data'
    sub_dir = 'query_auto_fill'
    dict_name = 'test_default_candi_query_lines.pt'
    dpath = os.path.join(data_home, sub_dir, dict_name)

    query_auto_fill = QueryAutoFill(dpath)
    input_query1 = '美国'
    input_query2 = '开车'


    def test_tokenizer(input_query):
        candi_query_list = query_auto_fill.forward(input_query)
        print('*--*' * 10)
        print('input_query:\t', input_query)
        print('candi_query:\t', candi_query_list, end='\n'*2)


    test_tokenizer(input_query1)
    test_tokenizer(input_query2)
