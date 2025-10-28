import torch
import torch.nn as nn

class ppiTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # Use batch_first inputs so tensors stay in [B, L, D] layout throughout.
        encoder_layer = nn.TransformerEncoderLayer(d_model=320, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc = nn.Linear(320, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb1, emb2):
        batch_size, _, embed_dim = emb1.shape
        # Insert a learned separator of zeros; match batch/device/dtype so cat works.
        gap_token = torch.zeros((batch_size, 1, embed_dim), device=emb1.device, dtype=emb1.dtype)
        concated = torch.cat((emb1, gap_token, emb2), dim=1)
        transformed = self.transformer_encoder(concated)
        x_pooled = transformed.mean(dim=1)
        x_out = self.fc(x_pooled)
        x_out = self.sigmoid(x_out)
        return x_out