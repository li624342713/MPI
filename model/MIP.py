import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MultiTextTransformer(nn.Module):
    def __init__(self,
                 vocab_sizes,
                 embed_dim=512,
                 num_heads=8,
                 num_layers=6,
                 num_classes=22,
                 max_seq_len=512):
        super().__init__()
        self.embedders = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(vocab_size, embed_dim),
                PositionalEncoding(embed_dim, max_len=max_seq_len)
            ) for vocab_size in vocab_sizes
        ])
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.encoders = nn.ModuleList([
            TransformerEncoder(encoder_layer, num_layers=num_layers)
            for _ in range(3)
        ])
        self.fusion_transformer = TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, inputs):
        encoded_features = []
        for i in range(3):
            embedding = self.embedders[i](inputs[i])
            embedding = embedding.permute(1, 0, 2)
            padding_mask = (inputs[i] == 0).transpose(0, 1)
            encoded = self.encoders[i](embedding, src_key_padding_mask=padding_mask)
            encoded_features.append(encoded.permute(1, 0, 2))
        fused = torch.cat(encoded_features, dim=1)
        if fused.dim() == 3:
            fused = fused.permute(1, 0, 2)
            padding_mask = torch.zeros(fused.size(0), fused.size(1),
                                       dtype=torch.bool, device=fused.device)
            fused = self.fusion_transformer(fused, src_key_padding_mask=padding_mask)
            fused = fused.permute(1, 0, 2)
            pooled = fused.mean(dim=1)
        else:
            pooled = fused
        logits = self.classifier(pooled)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_emb = nn.Embedding(max_len, d_model)
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_embed = self.position_emb(positions)
        return x + pos_embed