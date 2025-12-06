"""
Train logistic regression using sentence embeddings from the fine-tuned SentenceTransformer
"""

import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

# ============ Configuration ==================
INPUT_CSV = "../data/review_sentiment.csv"
SAVE_PATH = "../sentiment/models/sbert"
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
RANDOM_STATE = 42
NUM_LABELS = 3
TEST_SIZE = 0.2


MODEL_DIR = os.path.expanduser(
    "~/NLP/code/disneyland_review/sentiment/models"
)
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_OUT = os.path.join(MODEL_DIR, "logreg.joblib")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============ Load Data ==================

df = pd.read_csv(INPUT_CSV, encoding="latin1")
sentiment_map = {"neg": 0, "neu": 1, "pos": 2}
df["label"] = df["Sentiment"].map(sentiment_map)


X, y = df["Review_Text_bert"].tolist(), df["label"].tolist()

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Training size: {len(X_train)}, Validation size: {len(X_val)}")

# ====== Load Fine-Tuned Sentence Transformer =======

# Load fine-tuned sentence transformer
SAVE_PATH = os.path.expanduser(
    "~/NLP/code/disneyland_review/sentiment/models/sbert"
)
model = SentenceTransformer(SAVE_PATH)

# Generate sentence embedding
X_train_embed = model.encode(
    X_train,
    batch_size=BATCH_SIZE,
    convert_to_numpy=True,
    show_progress_bar=True
)

X_val_embed = model.encode(
    X_val,
    batch_size=BATCH_SIZE,
    convert_to_numpy=True,
    show_progress_bar=True
)

y_train_np = np.array(y_train)
y_val_np = np.array(y_val)

# Train LightGBM Classifier
print("\nTraining LightGBM classifier...")

lr_clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Trained on training data sentence embedding
lr_clf.fit(X_train_embed, y_train_np)

# Save model
joblib.dump(lr_clf, MODEL_OUT)

# =============== Evaluation ====================

# Predict on validation set
y_pred_val = lr_clf.predict(X_val_embed)

target_names = ["neg", "neu", "pos"]

print("Classification report:")
print(classification_report(y_val_np, y_pred_val, digits=4, target_names=target_names))

print("\nConfusion Matrix (Validation Data):")
print(confusion_matrix(y_val_np, y_pred_val))

print("\nF1 Macro (Validation Data):", f1_score(y_val_np, y_pred_val, average="macro"))
print("=" * 80)