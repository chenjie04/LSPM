import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from mlperf_compliance import mlperf_log


class Long_and_Short_term_Preference_Model(nn.Module):
    def __init__(self, nb_users, nb_items,
                 embed_dim,
                 mlp_layer_sizes, mlp_layer_regs):
        if len(mlp_layer_sizes) != len(mlp_layer_regs):
            raise RuntimeError('u dummy, layer_sizes != layer_regs!')
        if mlp_layer_sizes[0] % 2 != 0:
            raise RuntimeError('u dummy, mlp_layer_sizes[0] % 2 != 0')
        super(Long_and_Short_term_Preference_Model, self).__init__()
        nb_mlp_layers = len(mlp_layer_sizes)


        mlperf_log.ncf_print(key=mlperf_log.MODEL_HP_MF_DIM)

        # TODO: regularization?

        self.mlp_user_embed = nn.Embedding(nb_users, embed_dim)
        self.mlp_item_embed = nn.Embedding(nb_items, embed_dim)

        mlperf_log.ncf_print(key=mlperf_log.MODEL_HP_MLP_LAYER_SIZES, value=mlp_layer_sizes)
        self.mlp0 = nn.Linear(embed_dim*2,mlp_layer_sizes[0])
        self.mlp = nn.ModuleList()
        for i in range(1, nb_mlp_layers):
            self.mlp.extend([nn.Linear(mlp_layer_sizes[i - 1], mlp_layer_sizes[i])])  # noqa: E501


        self.history_embed = nn.Embedding(nb_items, embed_dim)

        self.lstm = nn.LSTM(embed_dim, embed_dim)
        self.W_s2 = nn.Linear(embed_dim, 256, bias=True)
        self.W_s1 = nn.Linear(256, 1, bias=True)

        self.merge = nn.Linear(mlp_layer_sizes[-1] * 2, mlp_layer_sizes[-1])
        self.final = nn.Linear(mlp_layer_sizes[-1], 1)


        self.mlp_user_embed.weight.data.normal_(0., 0.01)
        self.mlp_item_embed.weight.data.normal_(0., 0.01)

        def golorot_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6. / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)

        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features  # noqa: F841, E501
            limit = np.sqrt(3. / fan_in)
            layer.weight.data.uniform_(-limit, limit)
        for layer in self.mlp:
            if type(layer) != nn.Linear:
                continue
            golorot_uniform(layer)
        lecunn_uniform(self.final)

    def forward(self, user, item, history,sigmoid=False):

        #****************** Long-term preference module impleted by MLP module *****************************
        xmlpu = self.mlp_user_embed(user)
        xmlpi = self.mlp_item_embed(item)

        xmlp = torch.cat((xmlpu, xmlpi), dim=1)
        xmlp = self.mlp0(xmlp)
        for i, layer in enumerate(self.mlp):
            xmlp = layer(xmlp)
            xmlp = nn.functional.relu(xmlp)


        #******************************* Short-term preference module ***************************************
        #RNN module
        xhistory = self.history_embed(history)
        x_h_i = self.history_embed(item)

        xhistory = xhistory.transpose(0,1)
        lstm_out, lstm_hidden = self.lstm(xhistory)
        lstm_out = lstm_out.transpose(0,1)

        #attention module
        logits = self.W_s2(lstm_out)
        logits = torch.tanh(logits)
        logits = self.W_s1(logits)
        logits = torch.transpose(logits, 1, 2)
        weights = F.softmax(logits, -1)

        atnn_out = torch.bmm(weights, lstm_out)

        size = atnn_out.size()
        atnn_out = atnn_out.view(-1, size[-1])

        # MLP module
        rnn_out = torch.cat((atnn_out,x_h_i),dim=1)
        rnn_out = self.mlp0(rnn_out)
        for i, layer in enumerate(self.mlp):
            rnn_out = layer(rnn_out)
            rnn_out = nn.functional.relu(rnn_out)


        #*********************************** Output Module *****************************************************
        x = torch.cat((rnn_out, xmlp), dim=1)
        x = self.merge(x)
        x = F.relu(x)
        x = self.final(x)

        if sigmoid:
            x = torch.sigmoid(x)
        return x, weights
