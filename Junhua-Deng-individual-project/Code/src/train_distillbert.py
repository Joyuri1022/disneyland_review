#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune DistilBERT (or any Transformer model) for
Disneyland review rating prediction (1–5 stars).

This version does NOT require command-line arguments.
All hyperparameters and paths are set directly inside the script.
"""

import os
os.environ["WANDB_DISABLED"] = "true"
import numpy as np
import pandas as pd
import torch
from transformers import logging
logging.set_verbosity_info()


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)


# config

INPUT_CSV = "../data/review_clean.csv"
OUTPUT_DIR = "../models/distilbert_disneyland"

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2
RANDOM_STATE = 39
USE_FP16 = True

# load data
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the preprocessed dataset.
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, encoding="latin1")

    required_cols = ["Rating", "Review_Text_bert"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in CSV.")

    before = len(df)
    df = df.dropna(subset=["Rating", "Review_Text_bert"])
    print(f"Dropped {before - len(df)} rows with missing values.")

    df["Rating"] = df["Rating"].astype(int)
    df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5)]
    print(f"Using {len(df)} valid rows (rating 1–5).")

    return df




class ReviewsDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for (text, label) pairs.
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )
        encoded["labels"] = label
        return encoded



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")

    return {"accuracy": acc, "f1_macro": f1_macro}



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = load_data(INPUT_CSV)

    # Map Rating 1–5 -> labels 0–4
    df["label"] = df["Rating"] - 1

    texts = df["Review_Text_bert"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
    ).to(device)

    # Datasets
    train_dataset = ReviewsDataset(X_train, y_train, tokenizer, max_length=MAX_LENGTH)
    val_dataset = ReviewsDataset(X_val, y_val, tokenizer, max_length=MAX_LENGTH)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training Arguments

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,

        # tqdm ON
        disable_tqdm=False,

        save_total_limit=2,
        report_to="none",
        fp16=USE_FP16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating best checkpoint...")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete.")


if __name__ == "__main__":
    main()
