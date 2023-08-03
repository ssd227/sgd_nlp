# sgd_NLP

## 简介
    从0实现一些常见的NLP算法。
    ./python/sgd_nlp    库文件src
    ./apps              介绍sgd_nlp库的使用样例
    ./docs              对复杂概念或trick做展开说明
    ./docs/papers       nlp相关论文汇总

---
## 实现
### 1) Transformer (attention机制)
  - transformer
  - bert（待补充完整训练框架）

### 2) RNN
- gate control model
  - LSTM
  - GRU

### 3) Segmentation (分词相关)
  - 3.1基于词典的中文分词（概率统计模型）
    - max_match_tokenizer-最大匹配分词 
    - bi_max_match_tokenizer-双向最大匹配分词
    - max_porbability_path_tokenizer-基于语言模型的概率最大化分词
  - 3.2 query自动补全 
  - 3.3 热词检索（todo [参考blog](http://www.matrix67.com/blog/archives/5044)）
    
### 4) Embedding
  - glove
  - word2vec
  
### 5) LLM相关实践（fine tune相关）

---
## TODO
    - embedding （原理类似，不太想写了）
      - transE
      - transR




