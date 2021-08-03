import torch
import torch.nn as nn
import random
import numpy as np
import math
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# sent_size:句子中token数
# shift:论文中高斯自注意力机制 公式3 w
# bias:论文中高斯自注意力机制 公式3 b
# num_heads:注意力多头数
def Gaussian_Prior(sent_size, shift, bias, batch_size):
    Dis_M = np.zeros(shape=[sent_size, sent_size])  # 创建距离矩阵
    for i in range(sent_size):
        for j in range(sent_size):
            Dis_M[i][j] = (i - j) ** 2
    dis_M = torch.from_numpy(Dis_M)  # 转为tensor张量

    shift = torch.tensor(shift).unsqueeze(0).repeat(sent_size, 1).repeat(1, sent_size)
    bias = torch.tensor(bias).unsqueeze(0).repeat(sent_size, 1).repeat(1, sent_size)

    dis_M = - (shift * dis_M + bias)  # 论文中高斯自注意力机制 公式3 -|w * d^2 + b|
    result = dis_M.unsqueeze(0).repeat(batch_size, 1, 1)
    return result  # 输出的值为 高斯先验分布


# 克隆函数 起复制作用
def clones(module, N):  # The encoder is composed of a stack of  N=6 identical layers.
    # "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 每一个头 自注意力机制实现函数
def attention(query, key, value, mask=None, dropout=None):
    # "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# 多头注意力机制
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class BiLSTM(nn.Module):

    def __init__(self, lstm_hidden_dim, embed_dim, Gaussian_Prior, heads=8, sent_size=24, lstm_num_layers=2,
                 dropout=0.2):
        super(BiLSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embed_dim
        self.heads = heads  # 注意力头数
        self.sent_size = sent_size  # 句子中token个数
        self.gaussian = Gaussian_Prior  # 高斯先验分布

        # Bilstm
        self.bilstm = nn.LSTM(self.embed_dim, self.hidden_dim // 2, num_layers=lstm_num_layers, dropout=dropout,
                              bidirectional=True, bias=False, batch_first=True)

        # 多头注意力机制
        self.muti_attention = MultiHeadedAttention(h=heads, d_model=sent_size)

        self.layer = nn.Linear(lstm_hidden_dim, sent_size, bias=True)
        self.tanh = nn.Tanh()

        self.layer_1 = nn.Linear(sent_size, sent_size, bias=True)
        self.layer_2 = nn.Linear(sent_size, sent_size, bias=True)
        self.layer_3 = nn.Linear(sent_size, sent_size, bias=True)

    def forward(self, x):
        bilstm_out, (h, c) = self.bilstm(x)  # 得到bilstm的输出
        out = self.Gaussion_attention(bilstm_out)
        return out  # dimension:[batchsize * num_layers * hidden_dim]

    # 完整高斯自注意力机制 包含上面的高斯先验分布
    def Gaussion_attention(self, bilstm_out):
        new_bilstm_out = torch.DoubleTensor(bilstm_out.detach().cpu().numpy()).to(device)
        U_ij = self.tanh(self.layer(bilstm_out))  # 论文高斯自注意力机制 公式2
        query = self.layer_1(U_ij)
        key = self.layer_2(U_ij)
        value = self.layer_3(U_ij)
        U_ij = self.muti_attention(query, key, value)
        hidden = (U_ij + self.gaussian) / math.sqrt(U_ij.size(-1))  # 维度[batch_size, sent_size, sent_size]
        atten_score = torch.softmax(hidden, dim=-1)
        out = torch.matmul(atten_score, new_bilstm_out)  # 加权求和
        return out


"""
g = Gaussian_Prior(sent_size=500, shift=0.5, bias=0.5, batch_size=16).cuda()
model = BiLSTM(lstm_hidden_dim=300, embed_dim=784, Gaussian_Prior=g, heads=10, sent_size=500, lstm_num_layers=2,
               dropout=0.2).cuda()
data = torch.randn(16, 500, 784).cuda()

out = model(data)
print(out.shape)
"""