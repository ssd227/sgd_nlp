# encoding=utf8
from sgd_nlp.core.embedding.submodule.corpus_factory import CorpusFactoryGlove
from sgd_nlp.core.embedding.glove import Glove

from torch.optim.lr_scheduler import ExponentialLR
import torch
import os
import time
import pickle


# todo 需要提炼成 common function
def train(corpus_factory, model, optimizer, scheduler, weights_file, device, win_width):
    # train setting
    corpus_run_loop = 30  # 看n遍所有词对
    batch_size = 1024
    all_words_num = corpus_factory.word_pairs_num()  # 文档中的总词数
    epoch = int(corpus_run_loop * all_words_num / batch_size)  # 总共需要迭代几个epoch

    global_min_loss = 1e10

    # training loop
    for i in range(epoch):
        t1 = time.time()
        optimizer.zero_grad()

        # forward
        batch_data = corpus_factory.training_batch(batch_size, device=device)
        y = model.forward(batch_data)

        # objective function (loss function)
        j = torch.mean(y)  # minimize objective

        # backward and update weight
        j.backward()
        optimizer.step()

        # if epoch % 100 == 0:
        #     scheduler.step()

        # output info
        tmp_t = time.time() - t1
        if i % 100 == 0:
            print('epoch:{}/{}, loss:{}, csot_time: {}'
                  .format(i, epoch, j, tmp_t))

        # save best model
        if j < global_min_loss:
            global_min_loss = j
            torch.save(model.state_dict(), weights_file)
            print('new bset loss: {}'.format(j))


def load_glove_corpus(load_obj, obj_file_name, win_width, pair_symmetric):
    if load_obj and os.path.isfile(obj_file_name):
        with open(obj_file_name, 'rb') as fin:
            print("!!! load corpus factory success !!!")
            return pickle.load(fin)
    else:
        # corpus files path
        data_home = r'C:\Users\SGD\Desktop\sgd-代码库\sgd_deep_learning_framwork\sgd_nlp\data'
        sub_dir = r'friends\season10'
        corpus_dir = os.path.join(data_home, sub_dir)
        print('CURRENT PATH:\t', corpus_dir)

        # new obj from origin corpus file path
        corpus_factory = CorpusFactoryGlove(corpus_dir,
                                            win_width=win_width,
                                            pair_symmetric=pair_symmetric)
        corpus_factory.vocab.log_info()  # show log

        with open(obj_file_name, 'wb') as fout:
            pickle.dump(corpus_factory, fout)

        return corpus_factory


def app():
    # config
    device = torch.device('cuda')
    obj_file_name = 'save/glove/corpus_obj.cf'
    weights_file = "save/glove/glove_weights.path"
    win_width = 10
    emb_dim = 300

    sparse_emb = False
    pair_symmetric = True
    train_on_old = True
    load_weights = train_on_old  # 是否加载模型参数

    # class obj
    corpus_factory = load_glove_corpus(load_obj=train_on_old,
                                       obj_file_name=obj_file_name,
                                       pair_symmetric=pair_symmetric,
                                       win_width=win_width)

    model = Glove(emb_dim=emb_dim,
                  token_num=corpus_factory.token_num(),
                  sparse_emb=sparse_emb).to(device)

    for pa in model.parameters():
        print(pa)
        print(pa.device)

    # load weight
    if load_weights and os.path.isfile(weights_file):
        model.load_state_dict(torch.load(weights_file))
        print("!!! Load model weights success !!!")

    # optimizer
    # optimizer = torch.optim.SparseAdam(params=model.parameters(), lr=1e-1)
    optimizer = torch.optim.Adagrad(params=model.parameters(), lr=1e-6)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    train(corpus_factory=corpus_factory,
          model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          weights_file=weights_file,
          device=device,
          win_width=win_width, )


if __name__ == '__main__':
    app()
