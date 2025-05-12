from torch.utils.data import Dataset


class PegasusDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_input_len=512, max_output_len=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.summaries = summaries
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        summary = str(self.summaries[idx])

        # 编码输入文本
        input_enc = self.tokenizer(
            text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_enc = self.tokenizer(
            summary,
            max_length=self.max_output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": target_enc["input_ids"].squeeze()
        }