import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder

class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def get_datasets(csv_path, tokenizer_name='bert-base-uncased', max_len=128, batch_size=32):
        df = pd.read_csv(csv_path)
        texts = df["text"].values.astype(str)
        labels = LabelEncoder().fit_transform(df["label"])
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels,
            test_size=0.2,
            random_state=42
        )
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        train_dataset = BertDataset(
            texts=X_train,
            labels=y_train,
            tokenizer=tokenizer,
            max_len=max_len
        )
        val_dataset = BertDataset(
            texts=X_val,
            labels=y_val,
            tokenizer=tokenizer,
            max_len=max_len
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )
        return (train_loader, val_loader), tokenizer