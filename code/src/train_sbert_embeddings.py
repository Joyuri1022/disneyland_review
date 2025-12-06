"""
Fine-tune all-mpnet-base-v2 for sentence embeddings.
"""

import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ============ Configuration ==================
INPUT_CSV = "../data/review_sentiment.csv"
SAVE_PATH = "../sentiment/models/sbert"
PRETRAINED_MODEL_NAME = "all-mpnet-base-v2"
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
RANDOM_STATE = 42
NUM_LABELS = 3
TEST_SIZE = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========== Data Preparation =================
# Load data
df = pd.read_csv(INPUT_CSV, encoding="latin1")
# Map sentiment into numerical label
sentiment_map = {"neg": 0, "neu": 1, "pos": 2}
df["label"] = df["Sentiment"].map(sentiment_map)


X, y = df["Review_Text_bert"].tolist(), df["label"].tolist()

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ========== Sentence Transformer Fine Tune =================

# Convert training data into SBERT InputExample format
train_examples = [
    InputExample(texts=[text], label=int(label))
    for text, label in zip(X_train, y_train)
]

train_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# Load SBERT model
model = SentenceTransformer(PRETRAINED_MODEL_NAME, device=device)

# Instantiate loss functions
loss_fn = losses.BatchAllTripletLoss(model=model)

# Fine tune
model.fit(
    train_objectives=[(train_loader, loss_fn)],
    epochs=EPOCHS,
    warmup_steps=100,
    show_progress_bar=True,
    optimizer_params={'lr': 3e-5}, # Using your standard LR
)

# Save the fine-tuned SentenceTransformer model
os.makedirs(SAVE_PATH, exist_ok=True)
model.save(SAVE_PATH)
