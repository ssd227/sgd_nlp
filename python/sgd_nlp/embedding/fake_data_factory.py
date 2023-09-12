'''
only for testing
'''

import random
import torch

from ..common.vocab import Vocab


class FakeDataFactory:
    def __init__(self, row_num=1000, col_num=20, token_num=10000, neg_num=10):
        self.R = row_num
        self.C = col_num
        self.token_num = token_num
        self.neg_num = neg_num

        self.vocab = Vocab(list(range(0, self.token_num)))

    def yield_batch_data(self, batch_num, device):
        b_x = []
        for _ in range(batch_num):
            fake_sample = [random.randint(0, self.token_num) for _ in range(self.neg_num + 2)]
            b_x.append(fake_sample)

        return torch.tensor(b_x, dtype=torch.int64).to(device)


if __name__ == '__main__':
    factory = FakeDataFactory()
    batch_num = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(factory.yield_batch_data(100, device))
