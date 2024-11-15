import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding, CNN_Base

class CVAE(nn.Module):
    def __init__(self, args, data):
        super(CVAE, self).__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.input_dim = len(self.field_dims) * args.embed_dim
        self.text_embedding = nn.Linear(args.word_dim, args.embed_dim)
        self.cnn = CNN_Base( # 이미지 특성 추출
            input_size=(3, args.img_size, args.img_size),  # default: (3, 224, 224)
            channel_list=args.channel_list,                # default: [4, 8, 16]
            kernel_size=args.kernel_size,                  # default: 3
            stride=args.stride,                            # default: 2
            padding=args.padding,                          # default: 1
            batchnorm=args.cnn_batchnorm,                  # default: True
            dropout=args.cnn_dropout                       # default: 0.2
        )
        self.input_dim += torch.prod(torch.tensor(self.cnn.output_dim[1:]))
        self.input_dim += args.embed_dim * 2
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.latent_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x, images, users_text, books_text = x[0], x[1], x[2], x[3]

        # 피처 임베딩
        x = self.embedding(x)
        x = x.view(x.size(0), -1)

        # 이미지 특성 추출
        image_features = self.cnn(images)
        image_features = image_features.view(image_features.size(0), -1)

        # 텍스트 임베딩
        users_text_features = self.text_embedding(users_text)
        books_text_features = self.text_embedding(books_text)

        combined_input = torch.cat([x, image_features, users_text_features, books_text_features], dim=1)
        
        h = self.encoder(combined_input.float())
        mu, log_var = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
    
