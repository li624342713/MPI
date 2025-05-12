import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from dataset.BERTDataset import BertDataset


class RobertaDataset(BertDataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        super().__init__(texts, labels, tokenizer, max_len)

def get_datasets(csv_path, model_name='roberta-base', max_len=128, batch_size=32):
    df = pd.read_csv(csv_path)
    texts = df["text"].values.astype(str)
    labels = LabelEncoder().fit_transform(df["label"])
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42
    )
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
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