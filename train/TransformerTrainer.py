import torch
from transformers import AdamW, get_linear_schedule_with_warmup

# transformer 训练
class BertTrainer:
    def __init__(self, model, train_loader, val_loader, device, epochs=3):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                loss, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            self.evaluate()

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs[0]
                logits = outputs[1]
                total_loss += loss.item() * input_ids.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"\nValidation Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def save_model(self, path="output/best_model.pth"):
        torch.save(self.model.state_dict(), path)