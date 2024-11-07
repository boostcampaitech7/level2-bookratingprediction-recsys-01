import torch
import torch.nn as nn
from numpy import cumsum
from ._helpers import FeaturesLinear, FeaturesEmbedding, MLP_Base


class DeepFFM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.factor_dim = args.embed_dim
        self.feature_dim = sum(self.field_dims)
        self.num_fields = len(self.field_dims)

        self.linear = FeaturesLinear(self.field_dims, 1, bias=True)
        
        self.embedding = FeaturesEmbedding(self.field_dims, self.factor_dim)  
        self.ffm_embeddings = nn.ModuleList([
            nn.Embedding(self.feature_dim, self.factor_dim)
            for _ in range(self.num_fields)
        ])
        self.offsets = [0, *cumsum(self.field_dims)[:-1]]

        self.dnn = MLP_Base(
            input_dim=self.num_fields * self.factor_dim,
            embed_dims=args.mlp_dims,
            dropout=args.dropout,
            output_layer=False
        )

        self.interaction_dim = self.factor_dim * (self.num_fields * (self.num_fields - 1)) // 2

        final_input_dim = (args.mlp_dims[-1] + 
                          self.interaction_dim +  
                          1)  
        
        self.final_mlp = MLP_Base(
            input_dim=final_input_dim,
            embed_dims=[64, 32],
            dropout=args.dropout,
            output_layer=True
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x: torch.Tensor):

        first_order = self.linear(x).squeeze(1)
        

        embed_x = self.embedding(x)  
        deep_input = embed_x.view(-1, embed_x.size(1) * embed_x.size(2))
        deep_out = self.dnn(deep_input)
        

        x_ffm = x + x.new_tensor(self.offsets).unsqueeze(0)
        xv = [self.ffm_embeddings[f](x_ffm) for f in range(self.num_fields)]
        
        field_interactions = []
        for f in range(self.num_fields - 1):
            for g in range(f + 1, self.num_fields):
                field_interactions.append(xv[f][:, g] * xv[g][:, f])
        field_interactions = torch.cat(field_interactions, dim=1)

        combined_features = torch.cat([
            first_order.unsqueeze(1),
            field_interactions,
            deep_out
        ], dim=1)

        output = self.final_mlp(combined_features).squeeze(1)

        return output