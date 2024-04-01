import torchtext
from .config import data_home

trainds, validds, testds = torchtext.datasets.PennTreebank(root=data_home, split=('train', 'valid', 'test'))