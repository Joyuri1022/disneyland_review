#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DistilBERT Sentiment Classification (negative / neutral / positive)
with class_weight + F1 micro/macro + classification report

Transformers version: 4.56.1
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# ==============================================================
# Config
# ==============================================================

INPUT_CSV = "../data/review_clean.csv"
OUTPUT_DIR = "../models/distilbert_disneyland_sentiment"
MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2
RANDOM_STATE = 39
USE_FP16 = True


# ==============================================================
# Helper functions
# ==============================================================

def rating_to_sentiment(r: int) -> int:
    """Map original rating 1–5 to sentiment label 0/1/2."""
    if r <= 2:
        return 0  # negative
    elif r == 3:
        return 1  # neutral
    else:
        return 2  # positive


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin1")

    if "Rating" not in df.columns or "Review_Text_bert" not in df.columns:
        raise ValueError("CSV must contain Rating and Review_Text_bert columns.")

    df = df.dropna(subset=["Rating", "Review_Text_bert"])
    df["Rating"] = df["Rating"].astype(int)
    df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5)]

    df["sentiment_label"] = df["Rating"].apply(rating_to_sentiment)

    print("Sentiment distribution (0=neg,1=neu,2=pos):")
    print(df["sentiment_label"].value_counts().sort_index())

    return df


class ReviewsDataset(torch.utils.data.Dataset):
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

        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        tokens["labels"] = label
        return tokens


# ==============================================================
# Custom Trainer supporting class_weight
# ==============================================================

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """
        We ignore num_items_in_batch, but we must accept it,
        because Trainer will pass it in new transformers versions.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ==============================================================
# Metrics (F1 micro / macro)
# ==============================================================

def compute_metrics(eval_pred):
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


# ==============================================================
# Main
# ==============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = load_data(INPUT_CSV)

    texts = df["Review_Text_bert"].tolist()
    labels = df["sentiment_label"].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

    # ----------------------------------------------------------
    # class_weight
    # ----------------------------------------------------------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"class_weights: {class_weights}")

    # ----------------------------------------------------------
    # Tokenizer & model
    # ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    ).to(device)

    train_dataset = ReviewsDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = ReviewsDataset(X_val, y_val, tokenizer, MAX_LENGTH)

    data_collator = DataCollatorWithPadding(tokenizer)

    # ----------------------------------------------------------
    # TrainingArguments
    # ----------------------------------------------------------
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
        fp16=USE_FP16,
    )

    # ----------------------------------------------------------
    # Trainer with class_weight
    # ----------------------------------------------------------
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------
    print("Starting training...")
    trainer.train()

    # ----------------------------------------------------------
    # Evaluation (F1-micro, F1-macro)
    # ----------------------------------------------------------
    print("\n========== Evaluation ==========")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # ----------------------------------------------------------
    # Classification Report
    # ----------------------------------------------------------
    print("\n========== Classification Report ==========")

    pred_output = trainer.predict(val_dataset)
    preds = np.argmax(pred_output.predictions, axis=-1)
    labels_true = pred_output.label_ids

    target_names = ["negative", "neutral", "positive"]
    print(classification_report(
        labels_true,
        preds,
        target_names=target_names,
        digits=4
    ))
    print("============================================\n")

    # ----------------------------------------------------------
    # Save model
    # ----------------------------------------------------------
    print(f"Saving final model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
