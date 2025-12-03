#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeBERTa-v3-base sentiment classification with WeightedRandomSampler.
Adapted for ~12GB GPU (AMP + gradient checkpointing).

- Input CSV: ../data/review_clean.csv
- Required columns: "Rating" (1–5), "Review_Text_bert" (preprocessed text)
- Labels:
    0 = negative  (Rating 1–2)
    1 = neutral   (Rating 3)
    2 = positive  (Rating 4–5)
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    DebertaV2ForSequenceClassification
)

# ===============================
# Config
# ===============================

CSV_PATH = "../data/review_clean.csv"
OUTPUT_DIR = "../models/deberta_v3_base_disneyland"

# Use DeBERTa-v3-base
MODEL_NAME = "microsoft/deberta-v3-base"

# For 12GB GPU: keep max_length moderate and batch size small
MAX_LENGTH = 256
BATCH_SIZE = 12
EPOCHS = 3
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2
RANDOM_STATE = 39
USE_FP16 = True         # we will only enable fp16 when CUDA is available


# ===============================
# Helper functions
# ===============================

def rating_to_sentiment(r: int) -> int:
    """
    Map rating 1–5 to sentiment label 0/1/2.

    0: negative  (1–2 stars)
    1: neutral   (3 stars)
    2: positive  (4–5 stars)
    """
    if r <= 2:
        return 0
    elif r == 3:
        return 1
    else:
        return 2


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and create 'sentiment_label' column.
    Ensures ratings are in [1, 5] and drops missing values.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="latin1")

    required_cols = ["Rating", "Review_Text_bert"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV must contain column '{c}'")

    # Drop rows with missing rating or text
    df = df.dropna(subset=["Rating", "Review_Text_bert"])

    # Keep ratings between 1 and 5
    df["Rating"] = df["Rating"].astype(int)
    df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5)]

    # Create sentiment labels
    df["sentiment_label"] = df["Rating"].apply(rating_to_sentiment).astype(int)

    print("Sentiment distribution (0=neg,1=neu,2=pos):")
    print(df["sentiment_label"].value_counts().sort_index())
    print()

    return df


class ReviewsDataset(torch.utils.data.Dataset):
    """
    Simple dataset returning tokenized inputs + labels.
    """

    def __init__(self, texts, labels, tokenizer, max_length=256):
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
            max_length=self.max_length,
            padding=False,   # padding will be handled by DataCollatorWithPadding
        )
        encoded["labels"] = label
        return encoded


# ===============================
# Custom Trainer with sampler
# ===============================

class SamplerTrainer(Trainer):
    """
    Trainer subclass that overrides get_train_dataloader()
    to use a WeightedRandomSampler.
    """

    def __init__(self, train_labels=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store labels for computing sample weights
        self.train_labels = np.array(train_labels) if train_labels is not None else None

    def get_train_dataloader(self):
        """
        Build DataLoader using WeightedRandomSampler to handle class imbalance.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.train_labels is None:
            # Fallback: try reading labels from dataset if available
            if hasattr(self.train_dataset, "labels"):
                labels = np.array(self.train_dataset.labels)
            else:
                raise ValueError("No train_labels provided for SamplerTrainer.")
        else:
            labels = self.train_labels

        # Compute class counts and corresponding weights (1 / count)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        print("Train class counts:", class_counts)
        print("Train class weights (1/count):", class_weights)
        print()

        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),   # sample same number as training examples
            replacement=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )


# ===============================
# Metrics
# ===============================

def compute_metrics(eval_pred):
    """
    Compute accuracy, F1-micro, and F1-macro for evaluation.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }


# ===============================
# Main
# ===============================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load and prepare data
    df = load_data(CSV_PATH)
    texts = df["Review_Text_bert"].tolist()
    labels = df["sentiment_label"].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}\n")

    # Load tokenizer (slow tokenizer avoids some fast/SentencePiece issues)
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

    # Load model

    model = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
    )

    model.to(device)

    train_dataset = ReviewsDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = ReviewsDataset(X_val, y_val, tokenizer, MAX_LENGTH)

    data_collator = DataCollatorWithPadding(tokenizer)

    # Use fp16 only when CUDA is available (AMP)
    use_fp16_flag = bool(torch.cuda.is_available() and USE_FP16)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=20,
        disable_tqdm=False,
        save_total_limit=2,
        report_to="none",
        fp16=use_fp16_flag,                 # AMP on GPU
        gradient_accumulation_steps=2,      # effective batch size ~= 16
    )

    trainer = SamplerTrainer(
        train_labels=y_train,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --------------------------
    # Train
    # --------------------------
    print("Starting training with DeBERTa-v3-base + WeightedRandomSampler ...")
    trainer.train()

    # --------------------------
    # Evaluation
    # --------------------------
    print("\n========== Evaluation ==========")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print()

    # --------------------------
    # Classification report
    # --------------------------
    print("========== Classification Report ==========")
    pred_output = trainer.predict(val_dataset)
    preds = np.argmax(pred_output.predictions, axis=-1)
    true_labels = pred_output.label_ids

    target_names = ["negative", "neutral", "positive"]
    print(classification_report(
        true_labels,
        preds,
        target_names=target_names,
        digits=4
    ))
    print("===========================================\n")

    # --------------------------
    # Save model & tokenizer
    # --------------------------
    print(f"Saving final model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
