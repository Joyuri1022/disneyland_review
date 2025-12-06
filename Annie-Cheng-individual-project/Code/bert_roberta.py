"""
Fine tune BERT and RoBERTa for Disneyland review sentiment prediction (neg / neu / pos).
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ============ Configuration ==================

INPUT_CSV = "../data/review_sentiment.csv"
OUTPUT_ROOT = "../sentiment/models/"

MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 3e-5
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_FP16 = True

# candidate models
CANDIDATE_MODELS = [
    "bert-base-uncased",
    "roberta-base",
]


# ========== Data Preparation =================



class ReviewsDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch dataset class for tokenizing reviews
    """
    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )
        encoded["labels"] = int(self.labels[idx])
        return encoded


# ========== Compute Metrics =================

def compute_metrics(eval_pred):
    """
    Calculate evaluation metrics for Trainer
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# ========== Training Helper =================

def train_model(
    model_name: str,
    X_train,
    X_val,
    y_train,
    y_val,
    device: str = "cuda",
):
    """
    Helper function to train a sentiment classification model using HuggingFace Trainer.
    Trains the model, evaluates it on the validation set, and saves the best checkpoint.
    """

    # print("=" * 100)
    # print(f"Training model: {model_name}")
    # print("=" * 100)

    # create output dir for this model
    model_tag = model_name.replace("/", "_")
    output_dir = os.path.join(OUTPUT_ROOT, model_tag)
    os.makedirs(output_dir, exist_ok=True)


    # tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
    ).to(device)

    # datasets and collator
    train_dataset = ReviewsDataset(X_train, y_train, tokenizer, max_length=MAX_LENGTH)
    val_dataset = ReviewsDataset(X_val, y_val, tokenizer, max_length=MAX_LENGTH)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,

        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,

        save_total_limit=2,
        fp16=USE_FP16 and (device == "cuda"),
        report_to="none",
        disable_tqdm=True,
        seed=RANDOM_STATE,
        logging_dir=os.path.join(output_dir, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # train model
    trainer.train()

    # Evaluate the best checkpoints for the model
    metrics = trainer.evaluate()
    print(metrics)

    # Display classification report
    preds_output = trainer.predict(val_dataset)
    y_true = preds_output.label_ids
    y_pred = preds_output.predictions.argmax(axis=-1)

    target_names = ["neg", "neu", "pos"]
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
    )
    # print(cls_report)

    # Save model
    # print(f"Saving model and tokenizer to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Return relevant metrics and output directory
    result = {
        "model_name": model_name,
        "output_dir": output_dir,
        "eval_loss": metrics.get("eval_loss"),
        "eval_accuracy": metrics.get("eval_accuracy"),
        "eval_f1_macro": metrics.get("eval_f1_macro"),
    }

    del trainer, model
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


# ---------------- main ----------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")

    df = pd.read_csv(INPUT_CSV, encoding="latin1")

    sentiment_map = {"neg": 0, "neu": 1, "pos": 2}
    df["label"] = df["Sentiment"].map(sentiment_map)

    texts = df["Review_Text_bert"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_results = []

    # Loop through candidate models and train
    for model_name in tqdm(CANDIDATE_MODELS, desc="Training models"):
        result = train_model(
            model_name=model_name,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            device=device,
        )
        all_results.append(result)

    # sort by macro F1
    all_results.sort(key=lambda r: r["eval_f1_macro"], reverse=True)

    print("\nSummary Table:")
    for r in all_results:
        print(
            f"{r['model_name']}: "
            f"acc={r['eval_accuracy']:.4f}, "
            f"f1_macro={r['eval_f1_macro']:.4f}, "
            f"loss={r['eval_loss']:.4f}, "
            f"saved_at={r['output_dir']}"
        )

    # save comparison table to csv file
    # results_df = pd.DataFrame(all_results)
    # results_df.to_csv(os.path.join(OUTPUT_ROOT, "model_comparison.csv"), index=False)
    # print(f"\nSaved comparison table to {os.path.join(OUTPUT_ROOT, 'model_comparison.csv')}")


if __name__ == "__main__":
    main()

