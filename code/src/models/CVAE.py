import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding, CNN_Base

class CVAE(nn.Module):
    def __init__(self, args, data):
        super(CVAE, self).__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.input_dim = len(self.field_dims) * args.embed_dim
        self.cnn = CNN_Base(
            input_size=(3, args.img_size, args.img_size),  # default: (3, 224, 224)
            channel_list=args.channel_list,                # default: [4, 8, 16]
            kernel_size=args.kernel_size,                  # default: 3
            stride=args.stride,                            # default: 2
            padding=args.padding,                          # default: 1
            batchnorm=args.cnn_batchnorm,                  # default: True
            dropout=args.cnn_dropout                       # default: 0.2
        )
        self.input_dim += torch.prod(torch.tensor(self.cnn.output_dim[1:]))
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
        x, images = x[0], x[1]
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        image_features = self.cnn(images)
        image_features = image_features.view(image_features.size(0), -1)
        combined_input = torch.cat([x, image_features], dim=1)
        h = self.encoder(combined_input.float())
        mu, log_var = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
    
