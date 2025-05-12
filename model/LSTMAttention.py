import torch.nn as nn

# LSTM + attention
class LSTMTorchAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, num_heads=4, num_classes=22, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        query = lstm_out[:, -1, :].unsqueeze(1)
        attn_out, _ = self.attention(
            query=query,
            key=lstm_out,
            value=lstm_out
        )
        squeezed = attn_out.squeeze(1)
        output = self.dropout(squeezed)
        return self.fc(output)