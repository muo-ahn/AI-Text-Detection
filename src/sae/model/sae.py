import torch.nn as nn

class SAEClassifier(nn.Module):
    def __init__(self, input_dim=768, latent_dim=256, hidden_dims=[512, 256, 128]):
        super(SAEClassifier, self).__init__()
        
        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
        )

        # 디코더 (재구성)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )

        # 분류기
        # 기존 SAEClassifier의 Classifier → 더 깊은 구조
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2-class output (logits)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        out = self.classifier(z)
        return out, x_recon
