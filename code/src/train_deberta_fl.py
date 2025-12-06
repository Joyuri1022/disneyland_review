#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeBERTa-v3-small sentiment classification with:
- New label mapping: 1–2=neg, 3–4=neu, 5=pos
- Simple rule-based cleaning for obviously non-neutral 3-star reviews
- Neutral (and slightly negative) oversampling on train set
- WeightedRandomSampler for class balance
- Focal Loss for better handling of hard examples

Input CSV: ./data/review_clean.csv
Required columns:
    - "Rating" (int 1–5)
    - "Review_Text_bert" (preprocessed text)
Labels:
    0 = negative
    1 = neutral
    2 = positive
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# ===============================
# Config
# ===============================

CSV_PATH = "../data/review_clean.csv"
OUTPUT_DIR = "../models/deberta_disneyland_fl"

MODEL_NAME = "microsoft/deberta-v3-small"

MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2
RANDOM_STATE = 39
USE_FP16 = True

# Oversampling factors (on train set only)
NEG_OVERSAMPLE_FRAC = 0.5   # sample 0.5x negative (extra)
NEU_OVERSAMPLE_FRAC = 1.0   # sample 1.0x neutral (extra) -> roughly double neutral

# Focal loss hyperparameters
FOCAL_GAMMA = 2.0
# If you want per-class alpha, set a tensor of shape [num_classes], e.g. torch.tensor([1.0, 1.5, 1.0])
FOCAL_ALPHA = None  # keep None for now


# ===============================
# Label mapping / cleaning
# ===============================

def rating_to_sentiment(r: int) -> int:
    """
    Map rating 1–5 to sentiment label 0/1/2.

    0: negative  (1–2 stars)
    1: neutral   (3–4 stars)
    2: positive  (5 stars)
    """
    if r <= 2:
        return 0
    elif r <= 4:
        return 1
    else:
        return 2


# Simple lexicons to detect clearly positive/negative words
_STRONG_POS_WORDS = [
    "amazing", "awesome", "excellent", "fantastic", "incredible",
    "perfect", "love", "loved", "wonderful", "best", "great",
]
_STRONG_NEG_WORDS = [
    "terrible", "awful", "horrible", "worst", "disgusting",
    "hate", "hated", "nightmare", "useless", "awfully", "bad",
]


def is_strongly_positive(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _STRONG_POS_WORDS)


def is_strongly_negative(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _STRONG_NEG_WORDS)


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV, clean, and create sentiment labels.

    Steps:
    - Drop missing Rating/Review_Text_bert
    - Keep Rating in [1,5]
    - For rating==3, drop obviously very positive/negative texts
      (crude lexical filter, can be turned off if needed)
    - Map rating to 0/1/2 using rating_to_sentiment()
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="latin1")

    required_cols = ["Rating", "Review_Text_bert"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV must contain column '{c}'")

    df = df.dropna(subset=["Rating", "Review_Text_bert"])
    df["Rating"] = df["Rating"].astype(int)
    df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5)]

    # Optional: filter "obviously not neutral" 3-star reviews
    mask_3 = df["Rating"] == 3
    df_3 = df[mask_3].copy()
    to_drop_idx = []

    for idx, row in df_3.iterrows():
        text = str(row["Review_Text_bert"])
        if is_strongly_positive(text) or is_strongly_negative(text):
            to_drop_idx.append(idx)

    if to_drop_idx:
        print(f"Dropping {len(to_drop_idx)} obviously non-neutral 3-star reviews.")
        df = df.drop(index=to_drop_idx)

    df["sentiment_label"] = df["Rating"].apply(rating_to_sentiment).astype(int)

    print("Sentiment distribution after mapping (0=neg,1=neu,2=pos):")
    print(df["sentiment_label"].value_counts().sort_index())
    print()

    return df


# ===============================
# Dataset
# ===============================

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

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # padding handled by DataCollatorWithPadding
        )
        enc["labels"] = label
        return enc


# ===============================
# Focal Loss
# ===============================

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss implementation.

    Args:
        gamma: focusing parameter >= 0
        alpha: tensor of shape [num_classes] or float or None
        reduction: "none" | "mean" | "sum"
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        if isinstance(alpha, (list, tuple, np.ndarray)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha  # can be None or tensor
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, C)
        targets: (N,)
        """
        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1)
        # Pick the probability for the correct class
        pt = probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)  # (N,)

        # Compute log(pt)
        log_pt = torch.log(pt + 1e-12)

        # Compute alpha for each sample
        if self.alpha is not None:
            if self.alpha.dim() == 1:
                at = self.alpha.to(logits.device)[targets]
            else:
                at = self.alpha.to(logits.device)
        else:
            at = 1.0

        # Focal loss element-wise
        loss = -at * (1 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ===============================
# Custom Trainer with sampler + focal loss
# ===============================

class SamplerFocalTrainer(Trainer):
    """
    Trainer subclass that:
    - uses WeightedRandomSampler for train dataloader
    - applies Focal Loss instead of standard CE
    """

    def __init__(self, train_labels=None, focal_gamma=2.0, focal_alpha=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store labels for sampler
        self.train_labels = np.array(train_labels) if train_labels is not None else None

        # Create focal loss object
        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            reduction="mean",
        )

    def get_train_dataloader(self):
        """
        Build DataLoader using WeightedRandomSampler to handle class imbalance.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.train_labels is None:
            if hasattr(self.train_dataset, "labels"):
                labels = np.array(self.train_dataset.labels)
            else:
                raise ValueError("No train_labels provided for SamplerFocalTrainer.")
        else:
            labels = self.train_labels

        # Compute class counts and corresponding weights (1 / count)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        print("Train class counts (after oversampling):", class_counts)
        print("Train class weights for sampler (1/count):", class_weights)
        print()

        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
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

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """
        Override compute_loss to use focal loss.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.focal_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss


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

    # 1. Load and prepare data
    df = load_and_prepare_data(CSV_PATH)
    texts = df["Review_Text_bert"].tolist()
    labels = df["sentiment_label"].tolist()

    # 2. Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Original train size: {len(X_train)}, val size: {len(X_val)}")

    # 3. Oversample neutral (and a bit negative) on train set only
    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    df_neg = train_df[train_df["label"] == 0]
    df_neu = train_df[train_df["label"] == 1]
    df_pos = train_df[train_df["label"] == 2]

    aug_parts = [train_df]

    if NEU_OVERSAMPLE_FRAC > 0 and len(df_neu) > 0:
        neu_extra = df_neu.sample(
            frac=NEU_OVERSAMPLE_FRAC,
            replace=True,
            random_state=RANDOM_STATE,
        )
        aug_parts.append(neu_extra)

    if NEG_OVERSAMPLE_FRAC > 0 and len(df_neg) > 0:
        neg_extra = df_neg.sample(
            frac=NEG_OVERSAMPLE_FRAC,
            replace=True,
            random_state=RANDOM_STATE + 1,
        )
        aug_parts.append(neg_extra)

    train_df_aug = pd.concat(aug_parts).sample(
        frac=1.0, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    X_train_aug = train_df_aug["text"].tolist()
    y_train_aug = train_df_aug["label"].tolist()

    print(f"Augmented train size: {len(X_train_aug)}")
    print("Augmented train label distribution (0=neg,1=neu,2=pos):")
    print(train_df_aug["label"].value_counts().sort_index())
    print()

    # 4. Tokenizer & model
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
    )

    train_dataset = ReviewsDataset(X_train_aug, y_train_aug, tokenizer, MAX_LENGTH)
    val_dataset = ReviewsDataset(X_val, y_val, tokenizer, MAX_LENGTH)

    data_collator = DataCollatorWithPadding(tokenizer)

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

    trainer = SamplerFocalTrainer(
        train_labels=y_train_aug,
        focal_gamma=FOCAL_GAMMA,
        focal_alpha=FOCAL_ALPHA,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. Train
    print("Starting training with DeBERTa-v3-small + oversampling + WeightedRandomSampler + FocalLoss ...")
    trainer.train()

    # 6. Evaluation
    print("\n========== Evaluation ==========")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print()

    # 7. Classification report
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

    # 8. Save model & tokenizer
    print(f"Saving final model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
