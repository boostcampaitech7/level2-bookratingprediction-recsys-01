import torch
import torch.nn as nn
from ._helpers import FeaturesLinear, FeaturesEmbedding, FMLayer_Dense, MLP_Base
    

class NeuralFactorizationMachine(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.factor_dim = args.embed_dim
        
        self.linear = FeaturesLinear(self.field_dims, 1, bias=True)
        self.embedding = FeaturesEmbedding(self.field_dims, self.factor_dim)
        
        self.bi_pooling = FMLayer_Dense()
        
        self.dnn = MLP_Base(
            input_dim=self.factor_dim,
            embed_dims=[64, 32],
            dropout=0.2,
            output_layer=True
        )

    def forward(self, x: torch.Tensor):

        first_order = self.linear(x).squeeze(1)
        
        embed_x = self.embedding(x)
        
        bi_pooling = self.bi_pooling(embed_x)
        second_order =  bi_pooling.unsqueeze(1).expand(-1, self.factor_dim)

        dnn_out = self.dnn(second_order).squeeze(1)
        
        y = first_order + dnn_out
        
        return y