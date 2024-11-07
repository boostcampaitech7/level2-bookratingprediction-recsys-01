import torch
import torch.nn as nn
import numpy as np
from numpy import cumsum

    
    
#-----------------DCN+FFM MODEL----------------------#



# factorization을 통해 얻은 feature를 embedding 합니다.
# 사용되는 모델 : FM, CNN-FM, DCN
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims:list, embed_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = [0, *cumsum(field_dims)[:-1]]
        
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight.data, 0)  # cold-start

    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return self.embedding(x)  # (batch_size, num_fields, embed_dim)


# FM 계열 모델에서 활용되는 선형 결합 부분을 정의합니다.
# 사용되는 모델 : FM, FFM, WDN, CNN-FM
class FeaturesLinear(nn.Module):
    def __init__(self, field_dims:list, output_dim:int=1, bias:bool=True):
        super().__init__()
        self.feature_dims = sum(field_dims)
        self.output_dim = output_dim
        self.offsets = [0, *cumsum(field_dims)[:-1]]

        self.fc = nn.Embedding(self.feature_dims, self.output_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty((self.output_dim,)), requires_grad=True)
    
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight.data, 0)  # cold-start
            if isinstance(m, nn.Parameter):
                nn.init.constant_(m, 0)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return torch.sum(self.fc(x), dim=1) + self.bias if hasattr(self, 'bias') \
               else torch.sum(self.fc(x), dim=1)


#  MLP(initialize 제외)를 구현합니다.
# 사용되는 모델 : FFMwithDCN
class MLP_DCNwithFFM(nn.Module):
    def __init__(self, input_dim, embed_dims, 
                 batchnorm=True, dropout=0.2, output_layer=False):
        super().__init__()
        self.mlp = nn.Sequential()
        for idx, embed_dim in enumerate(embed_dims):
            self.mlp.add_module(f'linear{idx}', nn.Linear(input_dim, embed_dim))
            if batchnorm:
                self.mlp.add_module(f'batchnorm{idx}', nn.BatchNorm1d(embed_dim))
            self.mlp.add_module(f'relu{idx}', nn.ReLU())
            if dropout > 0:
                self.mlp.add_module(f'dropout{idx}', nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            self.mlp.add_module('output', nn.Linear(input_dim, 1))
        

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)
    
    
    

class FFMLayer_DCNwithFFM(nn.Module):

    def __init__(self, field_dims: list , embed_dim: int):
        super().__init__()
        self.num_fields = len(field_dims)
        self.feature_dim = sum(field_dims)
        self.embed_dim = embed_dim
        
        self.offsets = [0, *np.cumsum(field_dims)[:-1]]

        self.embeddings = torch.nn.ModuleList([
            nn.Embedding(self.feature_dim, self.embed_dim) for _ in range(self.num_fields)
        ])
        
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight.data, 0)  # cold-start

    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xv = [self.embeddings[f](x) for f in range(self.num_fields)]
        
        y = list()
        for f in range(self.num_fields - 1):
            for g in range(f + 1, self.num_fields):
                y.append(xv[f][:, g] * xv[g][:, f])
        y = torch.stack(y, dim=1)

        return torch.sum(y, dim=(2,1))
    
    
class CrossNetwork_DCNwithFFM(nn.Module):

    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

                                  
    def forward(self, x: torch.Tensor):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x