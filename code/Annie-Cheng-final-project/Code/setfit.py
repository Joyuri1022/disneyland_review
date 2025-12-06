from datasets import Dataset
from setfit import sample_dataset, SetFitModel, SetFitTrainer
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# 1. Load data
df = pd.read_csv("../data/review_sentiment.csv", encoding="latin1")
sentiment_map = {"neg": 0, "neu": 1, "pos": 2}
df["label"] = df["Sentiment"].map(sentiment_map)

TEXT_COLUMN = "Review_Text_bert"

X_full = df[TEXT_COLUMN].tolist()
y_full = df["label"].tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    random_state=42,
    stratify=y_full,
)

# 2. Hugging Face Datasets
train_dict = {"text": X_train, "label": y_train}
val_dict = {"text": X_val, "label": y_val}

hf_train = Dataset.from_dict(train_dict)
hf_val = Dataset.from_dict(val_dict)

# 3. Few shot sampling
NUM_SAMPLES_PER_CLASS = 32

sampled_train = sample_dataset(
    hf_train,
    num_samples=NUM_SAMPLES_PER_CLASS,
    label_column="label",
    seed=42,
)

print(sampled_train)
print(sampled_train["label"][:20])

# 4. Load SetFit model
model = SetFitModel.from_pretrained(
    "sentence-transformers/all-mpnet-base-v2",
)

# 5. Create trainer using old style API
trainer = SetFitTrainer(
    model=model,
    train_dataset=sampled_train,
    eval_dataset=hf_val,   # full validation
    metric="f1",
    batch_size=16,
    num_iterations=20,
    num_epochs=3,
)

# 6. Train
trainer.train()

save_dir = "../sentiment/models/setfit_fewshot"
os.makedirs(save_dir, exist_ok=True)
trainer.model.save_pretrained(save_dir)
print(f"Saved SetFit model to {save_dir}")

# 8. Detailed classification report using sklearn
print("\nPredicting on validation texts...")
y_pred = trainer.model.predict(X_val)

target_names = ["neg", "neu", "pos"]

print("\n" + "=" * 80)
print("Classification Report (SetFit, few shot)")
print("=" * 80)
print(classification_report(y_val, y_pred, target_names=target_names, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nF1 Macro:", f1_score(y_val, y_pred, average="macro"))
print("=" * 80)
