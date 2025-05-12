import torch
from dataset.Dataset import get_datasets
from model.LSTM import LSTMClassifier
from train.Trainer import Trainer

if __name__ == "__main__":
    # 数据集路径
    CSV_PATH = "dataset.csv"

    # 超参
    MAX_LEN = 200
    BATCH_SIZE = 64
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 64
    EPOCHS = 100

    torch.manual_seed(42)
    train_loader, val_loader, vocab = get_datasets(CSV_PATH, max_len=MAX_LEN, batch_size=BATCH_SIZE)

    # 模型, 选择适需要模型
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=22
    )

    # 训练，选择需要的训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    trainer.train(epochs=EPOCHS, lr=0.001)
    trainer.save_model()