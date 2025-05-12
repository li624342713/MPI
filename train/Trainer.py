import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def train(self, epochs=10, lr=0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.history["train_loss"].append(total_loss/len(self.train_loader))

            # 验证阶段
            val_loss, val_acc = self.evaluate()
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        self.plot_curves()

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
        return total_loss/len(self.val_loader), correct/len(self.val_loader.dataset)

    def plot_curves(self):
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(self.history["val_acc"], label="Val Acc", color="red")
        plt.legend()
        plt.show()

    def save_model(self, path="output/best_model.pth"):
        torch.save(self.model.state_dict(), path)