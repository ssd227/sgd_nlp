import torchtext
from .config import data_home

# 数据格式 (label:int, line:str)
train_iter, test_iter = torchtext.datasets.IMDB(root=data_home, split=('train','test'))

def tokenize(label, line):
    return line.split()

tokens = []
for label, line in train_iter:
    tokens += tokenize(label, line)