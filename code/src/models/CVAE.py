import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, args, data):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(args.num_items, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.latent_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.num_items)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
    
