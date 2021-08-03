import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    # TODO ---------------dataset------------------  datasets arguments
    parser.add_argument('--data_path', type=str, default='dataset/aclImdb/')
    # 数据文件路径
    parser.add_argument('--bert_model', type=str, default='pretrained_model/BERT_base_uncased')
    # BERT模型文件路径

    # TODO ---------------model--------------------- model arguments
    parser.add_argument('--embedding_dim', type=int, default=300)  # 对词做embedding后的词向量大小
    parser.add_argument('--mid_channels', type=int, default=300)  # 经过第一层CNN后 词的维度大小
    parser.add_argument('--out_channels', type=int, default=200)  # 经过第二层CNN后 词的维度大小
    parser.add_argument('--final_channels', type=int, default=36)  # 经过第三层CNN后 词的维度大小
    parser.add_argument('--lstm_hidden_dim', type=int, default=300)  # Bilstm中的隐藏层维度大小
    parser.add_argument('--vocabsize', type=int, default=30523)  # 词典大小
    parser.add_argument('--lstm_num_layers', type=int, default=2)  # Bilstm中的lstm层数
    parser.add_argument('--shift', type=float, default=0.5)   # 论文中高斯自注意力 公式3 w
    parser.add_argument('--bias', type=float, default=0.5)   # 论文中高斯自注意力 公式3 b
    parser.add_argument('--category', type=int, default=2)   # 结果分类数目

    # TODO ---------------training------------------ training arguments
    parser.add_argument('--lr', type=float, default=0.001)  # 学习率
    parser.add_argument('--wd', type=float, default=0)  # 衰减量
    parser.add_argument('--batch_size', type=int, default=16)  # batch size大小
    parser.add_argument('--epochs', type=int, default=10)  # 模型训练总轮数

    args = parser.parse_args()
    return args
