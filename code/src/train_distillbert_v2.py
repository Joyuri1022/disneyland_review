import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# =====================================
# Dataset
# =====================================
class TextClassificationDataset(Dataset):
    """Custom dataset for DistilBERT classification."""
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
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# =====================================
# Metrics
# =====================================
def compute_metrics(eval_pred):
    """Compute accuracy and macro F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
    }


# =====================================
# Trainer with class weights
# =====================================
class WeightedTrainer(Trainer):
    """Trainer that applies class weights in the loss function."""
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


# =====================================
# Main
# =====================================
def main():
    # Only keep CSV input and output directory
    parser = argparse.ArgumentParser(description="Weighted DistilBERT Training")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path.")
    parser.add_argument("--output", type=str, required=True, help="Where to save model.")
    args = parser.parse_args()

    # Fixed hyperparameters
    TEXT_COL = "Review_Text_bert"
    LABEL_COL = "Rating"
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    EPOCHS = 3
    LR = 5e-5
    MODEL_NAME = "distilbert-base-uncased"
    SEED = 39

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =====================================
    # Load data
    # =====================================
    df = pd.read_csv(args.input)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])

    texts = df[TEXT_COL].astype(str).tolist()
    labels = df[LABEL_COL].astype(int).tolist()
    labels = np.array(labels) - 1  # Convert 1–5 → 0–4

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=SEED,
    )

    print(f"Train size: {len(X_train)} | Validation size: {len(X_val)}")

    # =====================================
    # Class weights
    # =====================================
    num_labels = 5
    class_counts = np.bincount(y_train, minlength=num_labels)
    total = class_counts.sum()

    # w_c = N / (K * n_c)
    class_weights = total / (num_labels * class_counts)
    print("Class counts:", class_counts)
    print("Class weights:", class_weights)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # =====================================
    # Tokenizer + Model
    # =====================================
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    ).to(device)

    # =====================================
    # Datasets
    # =====================================
    train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = TextClassificationDataset(X_val, y_val, tokenizer, MAX_LENGTH)

    # =====================================
    # TrainingArguments (no evaluation_strategy to avoid error)
    # =====================================
    training_args = TrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=True,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,

        # These are broadly supported in older versions
        logging_steps=50,
        save_steps=500,
    )

    # =====================================
    # Trainer
    # =====================================
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )

    # =====================================
    # Train
    # =====================================
    trainer.train()

    # =====================================
    # Evaluation via trainer.evaluate()
    # =====================================
    print("\nRunning final evaluation with trainer.evaluate() ...")
    eval_results = trainer.evaluate(eval_dataset=val_dataset)
    print("\nEval metrics dict from trainer.evaluate():")
    print(eval_results)

    # =====================================
    # Detailed metrics via trainer.predict()
    # =====================================
    print("\nRunning detailed prediction on validation set...")
    preds_output = trainer.predict(val_dataset)
    y_pred = np.argmax(preds_output.predictions, axis=-1)

    # Convert back 0–4 → 1–5 for readability
    y_val_orig = y_val + 1
    y_pred_orig = y_pred + 1

    print("\nValidation Accuracy:", accuracy_score(y_val_orig, y_pred_orig))
    print("Validation F1-macro:", f1_score(y_val_orig, y_pred_orig, average="macro"))

    print("\nClassification Report:")
    print(classification_report(y_val_orig, y_pred_orig, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val_orig, y_pred_orig))

    # Also print the metrics returned by predict (usually test_* keys)
    print("\nMetrics dict from trainer.predict():")
    print(preds_output.metrics)

    # =====================================
    # Save model
    # =====================================
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nModel saved to: {args.output}")


if __name__ == "__main__":
    main()
