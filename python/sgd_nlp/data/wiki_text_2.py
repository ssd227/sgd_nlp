import torchtext
from .config import data_home

# 数据格式 (line:str)
trainds, validds, testds = torchtext.datasets.WikiText2(root=data_home, split=('train', 'valid', 'test'))
for x in testds:
    print(x)