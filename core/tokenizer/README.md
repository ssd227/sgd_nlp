# 分词相关问题


### 1、基于词典的中文分词（已完成、待测试）
    max_match_tokenizer.py
    模型1：最大匹配分词 
    
    bi_max_match_tokenizer.py
    模型2：双向最大匹配分词

    max_porbability_path_tokenizer.py
    模型3：基于语言模型的概率最大化分词

### 2、热词提取+热词发现（todo）
    原理：（from matrix64的博客）
    sadga

### 其他（todo）
    PS1：除了中文相关的应用，也可以用于英文词组提取
    PS2：bert出现以后，nlp模型直接使用word级别输入，分词的需求不是很大

    https://github.com/google/sentencepiece （现有可以使用的库, 学习借鉴一下）