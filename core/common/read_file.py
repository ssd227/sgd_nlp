import io
import os

# 遍历文件行
def yield_lines(file_path, reverse=False, encoding='utf-8'):
    # print(file_path)
    with io.open(file_path, encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0: continue
            if reverse: line = line[::-1]
            yield line


def yield_tokens_list(file_path, reverse=False, encoding='utf-8'):
    for line in yield_lines(file_path, reverse, encoding):
        yield line.split()


def yield_tokens(file_path, reverse=False, encoding='utf-8'):
    for line in yield_lines(file_path, reverse, encoding):
        for token in line.split():
            yield token


def yield_tokens_from_line(line, reverse=False, preprocess_func=None):
    return [preprocess_func(token) if preprocess_func else token for token in line.split()]

def yield_tokens_from_docs(doc_list, reverse=False, encoding='utf-8', preprocess_func=None):
    for doc in doc_list:
        for token in yield_tokens(doc, reverse=reverse, encoding=encoding):
            yield preprocess_func(token) if preprocess_func else token


def transform_tokens_list_to_tokens(tokens_list):
    for tokens in tokens_list:
        for token in tokens:
            yield token


def write_lines(lines, file_path):
    with open(file_path, 'w') as f:
        f.writelines(lines)


def write_tokens(tokens_list, file_path, split_c=' '):
    with open(file_path, 'w') as f:
        for tokens in tokens_list:
            f.write(split_c.join(tokens) + "\n")


if __name__ == '__main__':

    data_home = r"C:\Users\SGD\Desktop\sgd-代码库\sgd_deep_learning_framwork\sgd_nlp\data"
    sub_dir = 'tokenizer'
    dict_name = 'test_default_dict'

    x = os.path.join(data_home, sub_dir, dict_name)
    print(x)
    for item in yield_tokens(x, reverse=True):
        print(item)
