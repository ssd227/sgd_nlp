# sgd_NLP

## 简介
    NLP模型、算法从零实现
      * 涉及概率统计模型和deep learning模型
      * 包含分词、embedding、RNN、Transfomer等（扩充中）
      * 基于python 和 pytroch（autodiff）框架
      * 常用数据整理

---
## 目录
    ./apps              使用样例notebook
    ./docs              相关文档
    ./python/sgd_nlp    库文件
        ./data          语料加载
        ./embedding     emb算法
        ./rnn             
        ./tokenizer     中文分词
        ./transformer     

---
## 实现

### 1) Segmentation (分词相关)
  - 基于词典的中文分词（概率统计模型）
    - [X] max_match_tokenizer-最大匹配分词 [[code]](./python/sgd_nlp/tokenizer/max_match_tokenizer.py) [[app]](./apps/tokenizer/max_match_tokenizer.ipynb)
    - [X] bi_max_match_tokenizer-双向最大匹配分词 [[code]](./python/sgd_nlp/tokenizer/bi_max_match_tokenizer.py) [[app]](./apps/tokenizer/bi_max_match_tokenizer.ipynb)
    - [X] max_porbability_path_tokenizer-基于语言模型的概率最大化分词 [[code]](./python/sgd_nlp/tokenizer/max_probability_path_tokenizer.py) [[app]](./apps/tokenizer/max_probability_path_tokenizer.ipynb)
  - [X] query自动补全 [[app]](./apps/tokenizer/query_auto_fill.ipynb)
  - [ ] 热词检索（todo [参考](http://www.matrix67.com/blog/archives/5044)）
    
### 2) Embedding
  - [X] word2vec [[code]](./python/sgd_nlp/embedding/word2vec.py) [app: [skip_gram](./apps/embedding/train_w2v_skip_gram.ipynb) | [app-cbow](./apps/embedding/train_w2v_cbow.ipynb)]
  - [X] glove [[code]](./python/sgd_nlp/embedding/glove.py) [[app]](./apps/embedding/train_glove.ipynb)
  - [ ] transR、transE (原理类似w2v)

### 3) RNN
- gate control model
  - [X] LSTM [[code]](./python/sgd_nlp/rnn/lstm.py) [[app]](./apps/RNN/word_level_language_model.ipynb)
  - [X] GRU [[code]](./python/sgd_nlp/rnn/gru.py) [[app]](./apps/RNN/word_level_language_model.ipynb)

### 4) Transformer (attention机制)
  - [X] transformer [[code]](./python/sgd_nlp/transformer/transformer.py) [[app]](./apps/Transformer/transformer.ipynb)
  - [X] bert（待补充完整训练框架）[[code]](./python/sgd_nlp/transformer/bert.py) [[app]](./apps/Transformer/bert.ipynb)

### 5) Data (数据整理)
- [X] Ptb
- [X] wiki text-2
- [X] 语义分析数据集合 acl IMDB： 简单的二分类
- [X] SNLI数据集 ：standford natural language inference

## NLP其他常见应用
在./app相关文件下演示操作流程，不使用本库实现
- [ ] Part of speech tagging
- [ ] Bert Fine Tune 流程

---
## TODO
- 传统任务 Tagging、POS、实体识别、语法语义分析问题
  - 利用DL模型改写上述任务
- 概率统计模型 HMM、CRF、LDA主题模型
  - 目的：概率图模型的应用，实践MCMC、Gibs采样算法