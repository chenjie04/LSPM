import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextCF(nn.Module):
    def __init__(self,nb_users,nb_items,embed_dim):
        super(ContextCF, self).__init__()


        self.mf_user_embed = nn.Embedding(nb_users,embed_dim)
        self.mf_item_embed = nn.Embedding(nb_items,embed_dim)
        self.history_embed = nn.Embedding(nb_items,embed_dim)

        self.lstm = nn.LSTM(embed_dim,embed_dim)
        self.W_s2 = nn.Linear(embed_dim,256,bias=False)
        self.W_s1 = nn.Linear(256,1,bias=False)

        self.merge = nn.Linear(embed_dim * 2,embed_dim)
        self.final = nn.Linear(embed_dim,1)

    def forward(self, user, item, history, sigmoid=False):
        #MF model
        xmfu = self.mf_user_embed(user)
        xmfi = self.mf_item_embed(item)
        mf_out = xmfu * xmfi

        #RNN model
        xhistory = self.history_embed(history)

        x_h_i = self.history_embed(item)

        # lstm_out, lstm_hidden = self.lstm(xhistory)
        xhistory = xhistory.transpose(0,1)
        lstm_out, lstm_hidden = self.lstm(xhistory)
        lstm_out = lstm_out.transpose(0,1)

        logits = self.W_s2(lstm_out)

        logits = torch.tanh(logits)

        logits = self.W_s1(logits)

        logits = torch.transpose(logits,1,2)

        weights = F.softmax(logits,2)

        atnn_out = torch.bmm(weights, lstm_out)

        size = atnn_out.size()
        atnn_out = atnn_out.view(-1,size[-1])

        rnn_out = atnn_out * x_h_i

        x = torch.cat((mf_out,rnn_out),dim=1)

        x = self.merge(x)

        x = F.relu(x)

        x = self.final(x)

        if sigmoid:
            x = torch.sigmoid(x)

        return x

