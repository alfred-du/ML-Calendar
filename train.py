"""
Training script for calendar event summarization using T5-small.

Training was originally performed in Google Colab for GPU access.
This script documents the full training pipeline.
"""

import pandas as pd
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

def preprocess_function(examples, tokenizer):
    inputs = ["summarize: " + text for text in examples["event_text"]]
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        examples["summary"],
        max_length=16,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    # Load CSVs (not committed to repo)
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
    )

    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )

    trainer.train()

    model.save_pretrained("./calendar_summarizer")
    tokenizer.save_pretrained("./calendar_summarizer")

if __name__ == "__main__":
    main()
