import numpy as np
import torch
import torch.nn as nn

from mlperf_compliance import mlperf_log


class GMF(nn.Module):
    def __init__(self, nb_users, nb_items,
                 mf_dim, mf_reg):
        super(GMF, self).__init__()

        mlperf_log.ncf_print(key=mlperf_log.MODEL_HP_MF_DIM)

        # TODO: regularization?
        self.mf_user_embed = nn.Embedding(nb_users, mf_dim)
        self.mf_item_embed = nn.Embedding(nb_items, mf_dim)


        self.final = nn.Linear(mf_dim, 1)

        self.mf_user_embed.weight.data.normal_(0., 0.01)
        self.mf_item_embed.weight.data.normal_(0., 0.01)


        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features  # noqa: F841, E501
            limit = np.sqrt(3. / fan_in)
            layer.weight.data.uniform_(-limit, limit)

        lecunn_uniform(self.final)

    def forward(self, user, item, sigmoid=False):
        xmfu = self.mf_user_embed(user)
        xmfi = self.mf_item_embed(item)
        xmf = xmfu * xmfi

        x = self.final(xmf)
        if sigmoid:
            x = torch.sigmoid(x)
        return x
