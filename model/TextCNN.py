import torch
import torch.nn as nn
import torch.nn.functional as F

# TextCNN
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=22,
                 filter_sizes=(3, 4, 5), num_filters=100, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=num_filters,
                      kernel_size=(fs, embed_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    # 初始化
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.1)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            conv_out = conv_out.squeeze(3)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            pooled = pooled.squeeze(2)
            pooled_outputs.append(pooled)
        combined = torch.cat(pooled_outputs, 1)
        combined = self.dropout(combined)
        return self.fc(combined)