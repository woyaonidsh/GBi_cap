import torch
import torch.nn as nn

import config

from data import ACL_IMDB_data

from model import GBi_Cap
from model import CNN
from model import CapsNet
from model import Bilstm

from train import Trainer

argus = config.parse_args()


def main(argus):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('当前设备: ', device)

    # 加载数据
    train_data, test_data, train_size, test_size = ACL_IMDB_data.process_data(file=argus.data_path,
                                                                              batch_size=argus.batch_size,
                                                                              token_path=argus.bert_model)

    # 高斯先验
    guassion = Bilstm.Gaussian_Prior(sent_size=24, shift=argus.shift, bias=argus.bias,
                                     batch_size=argus.batch_size).to(device)

    # 构建模型
    CNN_model = CNN.CNN_net(mid_channels=argus.mid_channels, out_channels=argus.out_channels,
                            final_channels=argus.final_channels, embedding_dim=argus.embedding_dim,
                            vocabsize=argus.vocabsize)  # CNN模型

    CapsNet_model = CapsNet.CapsNet()  # 胶囊网络

    Bilstm_model = Bilstm.BiLSTM(lstm_hidden_dim=argus.lstm_hidden_dim, embed_dim=784,
                                 Gaussian_Prior=guassion)  # Bilstm模型

    model = GBi_Cap.GBi_Cap_model(CNN=CNN_model, CpasNet=CapsNet_model, Bilstm=Bilstm_model, category=argus.category,
                                  embed_dim=argus.embedding_dim).to(device)  # 总模型

    # 构建损失函数和优化器
    criterion = nn.BCELoss().to(device)  # 损失函数
    optimizer = torch.optim.Adam(params=model.parameters(), lr=argus.lr, weight_decay=argus.wd)  # 优化器

    # 构建训练器
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, batchsize=argus.batch_size, device=device,
                      datasize=train_size)

    # 开始训练
    for i in range(argus.epochs):
        loss = trainer.train(train_data=train_data)
        acc, f1 = trainer.test(test_data=test_data)
        print('Epoch: ', i, '  ', 'Loss: ', loss, 'F1: ', f1, 'Accuracy： ', acc)


if __name__ == "__main__":
    main(argus=argus)
