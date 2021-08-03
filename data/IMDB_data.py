"""
@incollection{SocherEtAl2013:RNTN,
title = {{Parsing With Compositional Vector Grammars}},
author = {Richard Socher and Alex Perelygin and Jean Wu and Jason Chuang and Christopher Manning and Andrew Ng and Christopher Potts},
booktitle = {{EMNLP}},
year = {2013}
}
"""

import pandas
import torch
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset


file = 'D:\Homework\Datasets\IMDB电影评论\labeledTrainData.tsv'
bert_token_path = 'D:\Homework\pretrained_model\BERT_base_uncased'


def stanford_data_process(filepath, token_path, batch_size):
    train_data = pandas.read_csv(filepath_or_buffer=filepath, sep='\t')  # 读入数据
    sentiments = []  # 情感标签
    reviews = []  # 文本
    tokenizer = BertTokenizer.from_pretrained(token_path)  # 加载BERT的分词器
    for data in tqdm(range(len(train_data['review'])), desc='split sentences into tokens'):
        tokens = tokenizer.encode(train_data['review'][data])   # 对每句话进行分词
        if len(tokens) < 500:   # token个数小于500的数据被保留
            for i in range(500 - len(tokens)):
                tokens.append(0)   # 长度不够500 补位元素0
            reviews.append(tokens)  # 将每句话的token保存到一个list中
            sentiments.append(train_data['sentiment'][data])   # 保存情感分类标签
    sum_datas = len(sentiments[0:21040])
    print('数据总数: ', sum_datas)   # 总数据条数
    sentiments = torch.tensor(sentiments[0:21040], dtype=torch.int64)  # 增加一个维度
    reviews = torch.tensor(reviews[0:21040], dtype=torch.int64)
    data_tensor = TensorDataset(sentiments, reviews)
    train_loader = DataLoader(dataset=data_tensor, batch_size=batch_size, shuffle=True)   # 装载数据
    return train_loader, sum_datas

