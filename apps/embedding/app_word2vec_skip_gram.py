# encoding=utf8

from sgd_nlp.core.embedding.submodule.corpus_factory import CorpusFactorySkipGram
from sgd_nlp.core.embedding.word2vec import SkipGram

from torch.optim.lr_scheduler import ExponentialLR
import torch
import os
import time
import pickle


# todo 需要提炼成 common function
def train(corpus_factory, model, optimizer, scheduler, weights_file, device, win_width, neg_k):
    # train setting
    corpus_run_loop = 5  # 看n遍文本
    batch_size = 2048
    all_words_num = corpus_factory.vocab.corpus_word_count  # 文档中的总词数
    epoch = int(corpus_run_loop * all_words_num / batch_size)  # 总共需要迭代几个epoch

    global_min_loss = 1e6

    # training loop
    for i in range(epoch):
        avg_time = 0
        t1 = time.time()

        optimizer.zero_grad()

        # forward
        batch_data = corpus_factory.training_batch(batch_num=batch_size,
                                                   device=device,
                                                   win_width=win_width,
                                                   neg_k=neg_k)
        y = model.forward(batch_data)

        # objective function (loss function)
        j_theta = torch.sum(y, dim=[1, 2]).mean()  # maximize objective
        nj_theta = -1 * j_theta  # minimize objective

        # backward and update weight
        nj_theta.backward()
        optimizer.step()

        if epoch % 20 == 0:
            scheduler.step()

        # output info
        tmp_t = time.time() - t1
        # avg_time = avg_time * 0.9 + 0.1 * tmp_t
        if i % 10 == 0:
            print('epoch:{}/{}, loss:{}, cost_time: {}'
                  .format(i, epoch, nj_theta, tmp_t))

        # save best model
        if nj_theta < global_min_loss:
            global_min_loss = nj_theta
            torch.save(model.state_dict(), weights_file)
            print('new bset loss: {}'.format(nj_theta))


# 获取语料数据
def load_corpus(load_obj=False):
    obj_file_name = 'save/skipgram/corpus_obj.cf'
    if load_obj and os.path.isfile(obj_file_name):
        with open(obj_file_name, 'rb') as fin:
            print("!!! load corpus factory success !!!")
            return pickle.load(fin)
    else:
        data_home = r'C:\Users\SGD\Desktop\sgd-代码库\sgd_deep_learning_framwork\sgd_nlp\data'
        sub_dir = r'friends\season10'
        corpus_dir = os.path.join(data_home, sub_dir)
        print('CURRENT PATH:\t', corpus_dir)

        corpus_factory = CorpusFactorySkipGram(corpus_dir)  # new obj from origin corpus file path
        corpus_factory.vocab.log_info()
        with open(obj_file_name, 'wb') as fout:
            pickle.dump(corpus_factory, fout)
        return corpus_factory


def app():
    # config
    train_on_old = True
    device = torch.device('cuda')
    load_weights = train_on_old  # 是否加载模型参数
    weights_file = "save/skipgram/skipgram_weights.path"
    win_width = 11  # context 窗口大小（前5-中间词-后5）
    neg_k = 15  # 负采样数
    emb_dim = 300


    # class obj
    corpus_factory = load_corpus(load_obj=train_on_old)
    model = SkipGram(emb_dim=emb_dim,
                     token_num=corpus_factory.token_num(),
                     sparse_emb=True).to(device)

    # load weight
    if load_weights and os.path.isfile(weights_file):
        model.load_state_dict(torch.load(weights_file))
        print("!!! Load model weights success !!!")

    # optimizer
    optimizer = torch.optim.SparseAdam(params=model.parameters(), lr=1e-1)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    train(corpus_factory=corpus_factory,
          model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          weights_file=weights_file,
          device=device,
          win_width=win_width,
          neg_k=neg_k, )


if __name__ == '__main__':
    app()

    """
    训练记录：
    optimizer = torch.optim.SparseAdam(params=w2v.parameters(), lr=1e-1)
    scheduler = ExponentialLR(optimizer, gamma=0.9) 10次一更新
        epoch:330/383, loss:0.021752286702394485, avg csot time:0.029899072647094727

    """
