import torch
import torch.nn as nn
from numpy import cumsum
from ._helpers import FeaturesLinear, MLP_Base


class NeuralFieldAwareFactorizationMachine(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.factor_dim = args.embed_dim
        self.feature_dim = sum(self.field_dims)
        self.num_fields = len(self.field_dims)

        self.linear = FeaturesLinear(self.field_dims, 1, bias=True)
        
        self.offsets = [0, *cumsum(self.field_dims)[:-1]]
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.feature_dim, self.factor_dim)
            for _ in range(self.num_fields)
        ])

        self.interaction_dim = self.factor_dim * (self.num_fields * (self.num_fields - 1)) // 2
        
        self.dnn = MLP_Base(
            input_dim=self.interaction_dim,
            embed_dims=[64, 32],
            dropout=0.2,
            output_layer=True
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x: torch.Tensor):

        first_order = self.linear(x).squeeze(1)
        
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xv = [self.embeddings[f](x) for f in range(self.num_fields)]
        
        field_interactions = []
        for f in range(self.num_fields - 1):
            for g in range(f + 1, self.num_fields):
                field_interactions.append(xv[f][:, g] * xv[g][:, f])

        field_interactions = torch.cat(field_interactions, dim=1)  # (batch_size, interaction_dim)
        
        dnn_out = self.dnn(field_interactions).squeeze(1)
        
        y = first_order + dnn_out
        
        return y