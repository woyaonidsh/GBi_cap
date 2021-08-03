import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GBi_Cap_model(nn.Module):
    def __init__(self, CNN, CpasNet, Bilstm, category, embed_dim):
        super(GBi_Cap_model, self).__init__()
        self.CNN = CNN  # CNN网络
        self.CpasNet = CpasNet  # 胶囊网络
        self.Bilstm = Bilstm  # Bilstm网络

        self.layer = nn.Linear(embed_dim * 24, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.CNN(x)
        logits, reconstruction = self.CpasNet(output.unsqueeze(1))
        out = self.Bilstm(reconstruction)
        result = self.MLP(out)
        return result

    def MLP(self, out):
        out = torch.FloatTensor(out.view(16, -1).contiguous().detach().cpu().numpy()).to(device)
        out = self.layer(out)
        out = self.sigmoid(out.squeeze(-1))
        return out


"""
from tqdm import tqdm
g = Bilstm.Gaussian_Prior(sent_size=24, shift=0.5, bias=0.5, batch_size=16).to('cuda:0')
cnn = CNN.CNN_net(mid_channels=300, out_channels=200, final_channels=36, embedding_dim=300, vocabsize=10000)
cpasnet = CapsNet.CapsNet()
bilstm = Bilstm.BiLSTM(lstm_hidden_dim=300, embed_dim=784, Gaussian_Prior=g)

model = GBi_Cap_model(CNN=cnn, CpasNet=cpasnet, Bilstm=bilstm, category=2, embed_dim=300).cuda()

datas = [_ for _ in range(16 * 500)]

datas = torch.tensor(datas, dtype=torch.int64).view(-1, 500).cuda()

for i in tqdm(range(21000)):
    out = model(datas)
"""