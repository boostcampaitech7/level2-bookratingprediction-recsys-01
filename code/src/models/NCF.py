import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding, CNN_Base, MLP_Base


# user와 item의 latent factor를 활용하여 GMF를 구현합니다.
# 그리고 MLP결과와 concat하여 NCF 모델을 구현하고 최종 결과를 도출합니다.
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.user_field_idx = [0]
        self.item_field_idx = [1]
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.mlp = MLP_Base(self.embed_output_dim, args.mlp_dims, args.batchnorm, args.dropout)
        self.fc = nn.Linear(32 + 64 + 128, 1)

        self.cnn = CNN_Base(
            input_size=(3, args.img_size, args.img_size),  # default: (3, 224, 224)
            channel_list=args.channel_list,                # default: [4, 8, 16]
            kernel_size=args.kernel_size,                  # default: 3
            stride=args.stride,                            # default: 2
            padding=args.padding,                          # default: 1
            batchnorm=args.batchnorm,                      # default: True
            dropout=args.cnn_dropout                       # default: 0.2
        )


    def forward(self, x):
        x, images = x[0], x[1]
        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        gmf = user_x * item_x

        x = self.mlp(x.view(-1, self.embed_output_dim))

        image_features = self.cnn(images)
        image_features = image_features.view(image_features.size(0), -1)

        #print(gmf.shape, x.shape, image_features.shape)

        x = torch.cat([gmf, x, image_features], dim=1)
        x = self.fc(x).squeeze(1)
        return x
