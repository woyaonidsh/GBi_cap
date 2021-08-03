from tqdm import tqdm
import torch
from utils import F1
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 训练器
class Trainer:
    def __init__(self, model, criterion, optimizer, batchsize, device, datasize):
        super(Trainer, self).__init__()
        self.batchsize = batchsize  # 梯度累计
        self.model = model  # 模型
        self.criterion = criterion  # 损失函数
        self.optimizer = optimizer  # 优化器
        self.epoch = 0  # 训练轮数
        self.device = device  # 设备名
        self.datasize = datasize  # 数据集大小

    # 训练函数
    def train(self, train_data):
        print('Start training model: ')
        self.optimizer.zero_grad()  # 梯度清零
        total_loss = 0.0  # 总损失

        for i, data in tqdm(enumerate(train_data), desc='Training epoch ' + str(self.epoch + 1) + ''):
            output = self.model(data[1].to(self.device))  # 模型的输出

            loss = self.criterion(output, data[0].to(self.device))  # 计算损失
            total_loss += loss.item()  # 累加损失
            loss.backward()  # 计算梯度
            if i % 64 == 0 and i != 0:   # 256笔数据更新一次参数
                self.optimizer.step()  # 更新参数
                self.optimizer.zero_grad()  # 清空梯度
                print('Loss is: %.4f' % (total_loss / i))  # 输出训练损失
        self.epoch += 1  # 训练epoch递增
        return total_loss / len(train_data)

    @torch.no_grad()
    def test(self, test_data):
        with torch.no_grad():
            print('Start test model: ')
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i, data in tqdm(enumerate(test_data), desc='Testing epoch ' + str(self.epoch) + ''):
                output = self.model(data[1].to(self.device))  # 模型的输出
#                out = torch.max(output, dim=-1)  # 确定输出的类别
                out = torch.gt(output, 0.5)
                for j in range(self.batchsize):
                    if out[j] == True and data[0][j] == 1:
                        TP += 1
                    if out[j] == False and data[0][j] == 0:
                        TN += 1
                    if out[j] == True and data[0][j] == 0:
                        FP += 1
                    if out[j] == False and data[0][j] == 1:
                        FN += 1
            acc, f1 = F1.F1_score(TP=TP, TN=TN, FP=FP, FN=FN)
            return acc, f1


"""
import torch.nn as nn
loss = nn.CrossEntropyLoss()
c = torch.tensor([0])
d = torch.tensor([[10, 100]], dtype=torch.float)

a = loss(d, c)
print(a)

import matplotlib.pyplot as plt

x = [0.02, 0.10, 0.13, 0.20, 0.25, 0.33, 0.45, 0.52, 0.55, 0.56, 0.57, 0.50, 0.52, 0.54]

Z = [0.3, 0.35, 0.38, 0.45, 0.49, 0.52, 0.54, 0.58, 0.63, 0.65, 0.70, 0.67, 0.68, 0.65]


pre = [0.1, 0.15, 0.18, 0.23, 0.24, 0.30, 0.32, 0.36, 0.42, 0.49, 0.53, 0.57, 0.51, 0.53]
recall = []

for i in range(14):
    recall.append((x[i] * pre[i]) / (2 * pre[i] - x[i]))
print(recall)

y = [_ for _ in range(len(Z))]

plt.plot(y, recall, linestyle = '--', label='recall')
plt.legend(['recall'])
plt.show()
"""