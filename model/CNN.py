import torch
import torch.nn as nn
import math
from torch.autograd import Variable


# 对token做embedding
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# 对token加入位置编码信息
class PositionalEncoding(nn.Module):
    # "Implement the PE function."
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


#  CNN网络
class CNN_net(nn.Module):
    def __init__(self, mid_channels, out_channels, final_channels, embedding_dim, vocabsize):
        super(CNN_net, self).__init__()
        # 第一层 CNN
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=mid_channels, kernel_size=3)
        # 第一层 池化层
        self.max_pooling1 = nn.MaxPool2d(kernel_size=3)
        # 第二层 CNN
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=out_channels, kernel_size=4)
        # 第二层 池化层
        self.max_pooling2 = nn.MaxPool2d(kernel_size=4)
        # 第三层 CNN
        self.conv3 = nn.Conv1d(in_channels=50, out_channels=final_channels, kernel_size=5)
        # 第三层 池化层
        self.max_pooling3 = nn.MaxPool2d(kernel_size=5)
        # 激活函数 RELU
        self.RELU = nn.ReLU()
        self.embedding = Embeddings(d_model=embedding_dim, vocab=vocabsize)   # embedding
        self.position = PositionalEncoding(d_model=embedding_dim)   # token的位置编码

    def forward(self, inputs):
        embedding = self.position(self.embedding(inputs))   # 将token变为向量
        embedding = embedding.permute(0, 2, 1)   # 维度置换
        # 第一层卷积网络
        out = self.conv1(embedding)
        out = self.max_pooling1(out)
        # 第二层卷积网络
        out = self.conv2(out)
        out = self.max_pooling2(out)
        # 第三层卷积网络
        out = self.conv3(out)
        out = self.RELU(out)
        return out


"""
from tqdm import tqdm
model = CNN_net(mid_channels=300, out_channels=200, final_channels=36, embedding_dim=300, vocabsize=3000).cuda()
inputs = torch.ones((16, 500), dtype=torch.int64).cuda()

for i in tqdm(range(21000)):
    out = model(inputs)
"""



