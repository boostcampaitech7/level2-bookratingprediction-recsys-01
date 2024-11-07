import torch
import torch.nn as nn
from ._helpers import FeaturesLinear, FeaturesEmbedding, MLP_DCNwithFFM, CrossNetwork_DCNwithFFM, FFMLayer_DCNwithFFM


class DCNwithFFM(nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.
    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017,
        baseline DeepCoNN 
    """

    def __init__(self, args, data):
        super().__init__()
        # COMMON
        self.field_dims = data['field_dims']
        self.ffm_linear = FeaturesLinear(self.field_dims, 1, bias=True)

        # FFM
        self.ffm = FFMLayer_DCNwithFFM(self.field_dims, args.ffm_embed_dim)

        # DCN
        self.dcn_embedding = FeaturesEmbedding(self.field_dims, args.dcn_embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.dcn_embed_dim
        self.cn = CrossNetwork_DCNwithFFM(self.embed_output_dim, args.cross_layer_num)
        self.mlp = MLP_DCNwithFFM(self.embed_output_dim, args.mlp_dims, args.batchnorm, args.dropout)
        self.cd_linear = nn.Linear(args.mlp_dims[-1], 1, bias=False)

        # final
        self.linear = nn.Linear(2, 1, bias=False)


    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # FFM
        ffm_term = self.ffm(x)
        ffm_li = self.ffm_linear(x)
        ffm_x = ffm_li + ffm_term.unsqueeze(1) # (batch_size, 1)
        
        # DCN - cross network
        embed_x = self.dcn_embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        x_out = self.mlp(x_l1)
        p = self.cd_linear(x_out)
        
        # concatenate
        p = self.linear(torch.cat([ffm_x,p], dim=1)) # (batch_size, 2) -> linear(2,1) -> (batch_size, 1)
        return p.squeeze(1)