import pandas as pd
from rouge_score import rouge_scorer
from dataset.PegasusDataset import PegasusDataset
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments


dataset = pd.read_csv("dataset_path")
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
train_dataset = PegasusDataset(dataset["train"]["article"], dataset["train"]["highlights"], tokenizer)
val_dataset = PegasusDataset(dataset["validation"]["article"], dataset["validation"]["highlights"], tokenizer)

# 配置训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./pegasus-finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
)

model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, label in zip(decoded_preds, decoded_labels):
        result = scorer.score(pred, label)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {k: sum(v) / len(v) for k, v in scores.items()}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
trainer.train()