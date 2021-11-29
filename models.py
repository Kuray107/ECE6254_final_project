import torch
import torch.nn as nn
import torch.nn.functional as F

import hparams

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_size, pool_size, cnn_dropout=0.0, normalize=None):
        super(ConvLayer, self).__init__()
        self.normalize=normalize
        if self.normalize == "batchnorm":
            self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding_size)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(cnn_dropout)
        self.maxpool = nn.MaxPool2d(pool_size)
        torch.nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        if self.normalize == "layernorm":
            x = F.layer_norm(x, (C,H,W))
        elif self.normalize == "batchnorm":
            x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)

        return x

class AttCNN(nn.Module):
    def __init__(self):
        super(AttCNN, self).__init__()
        self.num_convlayers = hparams.num_convlayers
        self.convlayers = nn.ModuleList()
        for i in range(self.num_convlayers):
            self.convlayers.append(ConvLayer(
                    in_channels = hparams.in_channels[i], 
                    out_channels = hparams.out_channels[i],
                    kernel_size = hparams.kernel_size[i],
                    padding_size = hparams.padding_size[i], 
                    pool_size = hparams.pool_size[i],
                    cnn_dropout=hparams.cnn_dropout,
                    normalize=hparams.normalize)
            )

        self.weight = nn.Parameter(torch.zeros(1, 1, 256))
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.decision = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dimension -> B*1*H*T
        for i, convlayer in enumerate(self.convlayers):
            x = self.convlayers[i](x)

        B, C, H, T = x.size()
        x = x.view(B, C*H, T).permute(0, 2, 1) # -> B, T, C*H

        attscores = (torch.sum(x*self.weight, dim=-1) + self.bias)/256 
        attscores = self.softmax(attscores).unsqueeze(1) # attscores.shape = B, 1, T
        weighted_sum_vector = torch.bmm(attscores, x).squeeze()
        preds = self.decision(weighted_sum_vector)
        preds = self.sigmoid(preds).squeeze() # B 

        return preds

class AttRNN(nn.Module):
    def __init__(self):
        super(AttRNN, self).__init__()
        self.num_convlayers = hparams.num_convlayers
        self.convlayers = nn.ModuleList()
        for i in range(self.num_convlayers):
            self.convlayers.append(ConvLayer(
                    in_channels = hparams.in_channels[i], 
                    out_channels = hparams.out_channels[i],
                    kernel_size = hparams.kernel_size[i],
                    padding_size = hparams.padding_size[i], 
                    pool_size = hparams.pool_size[i],
                    cnn_dropout=hparams.cnn_dropout,
                    normalize=hparams.normalize)
            )

        self.rnn = Bi_LSTM(256, 64, dropout=hparams.rnn_dropout, num_layers=hparams.num_rnnlayers)
        self.attention = nn.Linear(64, 64)
        self.att_dropout = nn.Dropout(hparams.att_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.decision = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dimension -> B*1*H*T
        for i, layer in enumerate(self.convlayers):
            x = self.convlayers[i](x)

        B, C, H, T = x.size()
        x = x.view(B, C*H, T).permute(0, 2, 1) # -> B, T, C*H
        x = self.rnn(x) # B, T, 64

        query = self.attention(x[:, -1, :]).unsqueeze(2) # query.shape = B, 64, 1
        query = self.att_dropout(query) 
        attscores = torch.bmm(x, query).permute(0, 2, 1) # B, 1, T
        attscores = self.softmax(attscores)
        weighted_sum_vector = torch.bmm(attscores, x).squeeze()
        preds = self.decision(weighted_sum_vector)
        preds = self.sigmoid(preds).squeeze() # B 

        return preds


class Bi_LSTM(nn.Module):

    def __init__(self, n_in, n_out, dropout=0, num_layers=1):
        super(Bi_LSTM, self).__init__()
        self.rnn = nn.LSTM(n_in, n_out // 2, bidirectional=True, batch_first=True,
                           dropout=dropout, num_layers=num_layers)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        return recurrent
       
class Bi_GRU(nn.Module):

    def __init__(self, n_in, n_out, dropout=0, num_layers=1):
        super(Bi_GRU, self).__init__()

        self.rnn = nn.GRU(n_in, n_out//2, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        return recurrent


