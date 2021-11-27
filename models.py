import torch
import torch.nn as nn
import torch.nn.functional as F

import hparams

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, normalize=None):
        super(ConvLayer, self).__init__()
        self.normalize=normalize
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.maxpool = nn.MaxPool2d(pool_size)
        torch.nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.normalize == "layernorm":
            x = F.layer_norm(x, (C,H,W))
        elif self.normalize == "batchnorm":
            x = nn.BatchNorm2d(num_features=C)(x)

        x = self.conv(x)
        x = self.maxpool(x)

        return x


class AttRNN(nn.Module):
    def __init__(self):
        super(AttRNN, self).__init__()
        self.conv1 = ConvLayer(in_channels=1, out_channels=16,
                kernel_size = (3, 3), pool_size = (2, 2), 
                normalize=hparams.normalize)

        self.conv2 = ConvLayer(in_channels=16, out_channels=4,
                kernel_size = (2, 3), pool_size = (2, 1),
                normalize=hparams.normalize)

        self.rnn1 = Bi_LSTM(4*24, 64)
        self.rnn2 = Bi_LSTM(64, 64)
        self.attention = nn.Linear(64, 64)
        self.decision = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, normalization=None):
        x = x.unsqueeze(1) # B*1*H*T
        #print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)


        B, C, H, T = x.size()
        #print(x.size())
        x = x.view(B, C*H, T).permute(0, 2, 1)
        x = self.rnn1(x)
        x = self.rnn2(x).contiguous() # B, T, 64
        #print(x.size())

        query = self.attention(x[:, -1, :]).unsqueeze(2) 
        attscores = torch.bmm(x, query).permute(0, 2, 1) # B, T, 1
        weighted_sum_vector = torch.bmm(attscores, x).squeeze()
        preds = self.decision(weighted_sum_vector)
        preds = self.softmax(preds)

        return preds






        






class Bi_LSTM(nn.Module):

    def __init__(self, n_in, n_out, dropout=0, num_layers=1):
        super(Bi_LSTM, self).__init__()
        self.rnn = nn.LSTM(n_in, n_out // 2, bidirectional=True, batch_first=True,
                           dropout=dropout, num_layers=num_layers)
        #self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        #b, T, h = recurrent.size()
        #t_rec = recurrent.contiguous().view(b * T, h)

        #output = self.embedding(t_rec)  # [T * b, nOut]
        #output = output.view(b, T, -1)
        return recurrent
       
class BidirectionalGRU(nn.Module):

    def __init__(self, n_in, n_out, dropout=0, num_layers=1):
        super(BidirectionalGRU, self).__init__()

        self.rnn = nn.GRU(n_in, n_out, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        return recurrent
