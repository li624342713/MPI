import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import re


# 构建数据集，需要自动注入
class PatentDataset(Dataset):
    def __init__(self, texts, labels, word_index, max_len=200):
        self.texts = texts
        self.labels = labels
        self.word_index = word_index
        self.max_len = max_len
        self.unk_index = 1  # 未知词索引

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 清洗和分词
        tokens = re.sub(r"[^\w\s]", "", text.lower()).split()[:self.max_len]
        # 转换为索引
        indices = [self.word_index.get(token, self.unk_index) for token in tokens]
        # 填充
        padded = indices + [0] * (self.max_len - len(indices))
        return torch.tensor(padded), torch.tensor(self.labels[idx])


def build_vocab(texts, max_words=21128 + 1):
    word_counts = defaultdict(int)
    for text in texts:
        for word in re.sub(r"[^\w\s]", "", text.lower()).split():
            word_counts[word] += 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    word_index = {"<pad>": 0, "<unk>": 1}
    for idx, (word, _) in enumerate(sorted_words[:max_words - 2]):
        word_index[word] = idx + 2
    return word_index

# 数据集获取
def get_datasets(csv_path, max_len=200, batch_size=64):
    df = pd.read_csv(csv_path)
    texts = df["text"].values
    labels = LabelEncoder().fit_transform(df["label"])
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    word_index = build_vocab(X_train)
    train_loader = DataLoader(
        PatentDataset(X_train, y_train, word_index, max_len),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        PatentDataset(X_val, y_val, word_index, max_len),
        batch_size=batch_size
    )
    return train_loader, val_loader, word_index