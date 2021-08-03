import os
import torch
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

file = '../dataset/aclImdb/'
vocab_file = '../dataset/aclImdb/imdb.vocab'
batch_size = 16
token_path = '../pretrained_model/BERT_base_uncased'


def process_data(file, batch_size, token_path):
    train_sentiments = []
    train_data = []  # 训练数据集
    test_data = []  # 测试数据集
    test_sentiments = []
    tokenizer = BertTokenizer.from_pretrained(token_path)  # 分词器
    train_sentimens = ['train/neg/', 'train/pos/']  # 文件夹
    for sentimen in train_sentimens:
        file_path = file + sentimen
        for text in tqdm(os.listdir(file_path), desc='Reading training data: '):
            with open(file_path + text, 'r', encoding='utf-8') as txt:
                tokens = tokenizer.encode(txt.read())    # 分词
                if len(tokens) < 500:   # token数小于500的数被保留
                    for i in range(500 - len(tokens)):
                        tokens.append(0)
                    train_data.append(tokens)    # 添加分词后的token
                    if int(text[-5]) < 5:  # 情感划分, 小于5为负类
                        train_sentiments.append(0)
                    else:
                        train_sentiments.append(1)
    train_data = torch.tensor(train_data[0:(len(train_data)//batch_size) * batch_size], dtype=torch.int64)
    train_sentiments = torch.tensor(train_sentiments[0:(len(train_sentiments)//batch_size) * batch_size],
                                    dtype=torch.float)
    train_tensor = TensorDataset(train_sentiments, train_data)
    train_loader = DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=True)   # 装载训练数据

    test_sentmens = ['test/neg/', 'test/pos/']
    for sentimen in test_sentmens:
        file_path = file + sentimen
        for text in tqdm(os.listdir(file_path), desc='Reading test data: '):
            with open(file_path + text, 'r', encoding='utf-8') as tst:
                tokens = tokenizer.encode(tst.read())
                if len(tokens) < 500:
                    for i in range(500 - len(tokens)):
                        tokens.append(0)
                    test_data.append(tokens)   # 添加分词后的token
                    if int(text[-5]) < 5:
                        test_sentiments.append(0)
                    else:
                        test_sentiments.append(1)
    test_data = torch.tensor(test_data[0:(len(test_data)//batch_size) * batch_size], dtype=torch.int64)
    test_sentiments = torch.tensor(test_sentiments[0:(len(test_sentiments)//batch_size) * batch_size],
                                   dtype=torch.float)
    test_tensor = TensorDataset(test_sentiments, test_data)
    test_loader = DataLoader(dataset=test_tensor, batch_size=batch_size)   # 装载测试数

    train_sum = (len(train_data) // batch_size) * batch_size
    test_sum = (len(test_data) // batch_size) * batch_size
    print('训练集大小: ', train_sum)
    print('测试集大小: ', test_sum)
    return train_loader, test_loader, train_sum, test_sum

# a , b, c, d = process_data(file=file, vocab_file=vocab_file, batch_size=batch_size, token_path=token_path)
