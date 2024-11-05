import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding

class CVAE(nn.Module):
    def __init__(self, args, data):
        super(CVAE, self).__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.input_dim = len(self.field_dims) * args.embed_dim
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
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        h = self.encoder(x.float())
        mu, log_var = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
    
