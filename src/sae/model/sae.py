import torch.nn as nn

class SAEClassifier(nn.Module):
    def __init__(self, input_dim=768, latent_dim=128, hidden_dims=[512, 256, 128]):
        super(SAEClassifier, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[1], latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        out = self.classifier(z)
        return out, x_recon
