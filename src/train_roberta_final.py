"""
Fine tune RoBERTa for Disneyland review sentiment prediction (neg / neu / pos).
Uses custom class-weighted loss with base_weights = [1.0, 1.4, 0.8], normalized by their mean.
"""

import os
os.environ["WANDB_DISABLED"] = "true"
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ============== Configuration ===============
INPUT_CSV = "../data/review_sentiment.csv"
OUTPUT_ROOT = "../sentiment/models/"


MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 3e-5
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_FP16 = True
NUM_LABELS = 3

# models to compare
CANDIDATE_MODELS = [
    "roberta-base",
]

# ============== Data Preparation ==============

class ReviewsDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch dataset for tokenizing text reviews.
    Each item returns tokenized input IDs, attention masks, and the corresponding label.
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


# ========== Compute Metrics ==============

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for HuggingFace Trainer.
    Return:
         accuracy
         macro F1 score
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# ========== Custom Weighted Trainer ===========
class WeightedTrainer(Trainer):
    """
    Custom HuggingFace Trainer that applies class-weighted cross-entropy loss
    to handle class imbalance during fine-tuning.
    """
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
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
        Computes loss using class-weighted cross entropy
        """
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        labels = labels.long()
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if return_outputs:
            return loss, outputs
        return loss



# ========== Training Helper ===============

def train_model(
    model_name: str,
    X_train,
    X_val,
    y_train,
    y_val,
    class_weights: torch.Tensor,
    device: str = "cuda",
):
    """
    Helper function to fine tunes a sentiment classification model using HuggingFace Trainer.
    Performs training, evaluates on the validation set, saves the best checkpoint,
    and returns key performance metrics.
    """

    # Create output directory
    model_tag = model_name.replace("/", "_")
    output_dir = os.path.join(OUTPUT_ROOT, f"{model_tag}_weighted")
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and model architecture
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
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

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )

    t0 = time.time()
    trainer.train()
    print("Training Duration (secs):", round(time.time() - t0, 2))

    # Evaluate best checkpoint
    metrics = trainer.evaluate()
    print(metrics)

    # Generate classification reports
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


    # Save model
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
    print(f"Using device: {device}")

    df = pd.read_csv(INPUT_CSV, encoding="latin1")

    # Converts sentiment into numerical labels
    sentiment_map = {"neg": 0, "neu": 1, "pos": 2}
    df["label"] = df["Sentiment"].map(sentiment_map)

    texts = df["Review_Text_bert"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    # split data into train/validation set
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


    # Define custom class weights
    base_weights = np.array([1.0, 1.4, 0.8])
    class_weights = base_weights / base_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


    for model_name in tqdm(CANDIDATE_MODELS, desc="Training models"):
        result = train_model(
            model_name=model_name,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            class_weights=class_weights,
            device=device,
        )


if __name__ == "__main__":
    main()
