import torch
import torch.nn as nn
import torch.nn.functional as F

class ppiTransformer(nn.Module):
    def __init__(self):
        super(ppiTransformer, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=320, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.fc = nn.Linear(320, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb1, emb2):
        concated = torch.cat((emb1, torch.zeros((1, 1, 320)), emb2), dim=1)
        transformed = self.transformer_encoder(concated)
        x_pooled = transformed.mean(dim=1)
        x_out = self.fc(x_pooled)
        x_out = self.sigmoid(x_out)
        return x_out