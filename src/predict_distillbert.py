#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Use the fine-tuned DistilBERT model to predict Disneyland review ratings (1–5 stars).

Only supports 3 branches:
    - Disneyland_California
    - Disneyland_Paris
    - Disneyland_HongKong
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================================
# 🔧 CONFIG
# =========================================

MODEL_DIR = "../models/distilbert_disneyland"
MAX_LENGTH = 256


# =========================================
# 📌 NORMALIZE BRANCH (only 3 allowed)
# =========================================

def normalize_branch(user_input: str):
    """
    Normalize user-provided branch name into one of:
        - Disneyland_California
        - Disneyland_Paris
        - Disneyland_HongKong
    """

    s = user_input.strip().lower().replace(" ", "").replace("_", "")

    if "california" in s or "ca" in s:
        return "Disneyland_California"

    if "paris" in s or "france" in s:
        return "Disneyland_Paris"

    if "hong" in s or "hk" in s or "hongkong" in s:
        return "Disneyland_HongKong"

    # default fallback: Invalid → return None
    return None


# =========================================
# 📌 MODEL LOADING
# =========================================

def load_model_and_tokenizer(model_dir: str = MODEL_DIR):
    """
    Load fine-tuned model + tokenizer.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory '{model_dir}' does not exist. "
            f"Run train_distilbert.py first."
        )

    print(f"Loading model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Using device: {device}")
    return tokenizer, model, device


# =========================================
# 📌 INFERENCE
# =========================================

def predict_rating(review_text, branch, tokenizer, model, device):
    """
    Predict rating (1–5 stars) for a single review & branch.
    """
    full_text = f"Branch: {branch}. Review: {review_text}"

    encoded = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs_tensor = F.softmax(logits, dim=-1).squeeze(0)
        probs = probs_tensor.cpu().tolist()
        pred_label = int(torch.argmax(probs_tensor).item())
        pred_rating = pred_label + 1  # map 0–4 → 1–5

    return pred_rating, probs


# =========================================
# 🧪 INTERACTIVE LOOP
# =========================================

def main():
    tokenizer, model, device = load_model_and_tokenizer(MODEL_DIR)

    print("\n=== Disneyland Rating Prediction (3 branches only) ===")
    print("Supported branches:")
    print("  - California")
    print("  - Paris")
    print("  - HongKong / HK")
    print("Type 'quit' to exit.\n")

    while True:
        raw_branch = input("Branch (California / Paris / HongKong): ").strip()
        if raw_branch.lower() == "quit":
            break

        branch = normalize_branch(raw_branch)

        if branch is None:
            print("❌ Invalid branch. Please enter California / Paris / HongKong.\n")
            continue

        review_text = input("Enter review text: ").strip()
        if review_text.lower() == "quit":
            break

        rating, probs = predict_rating(
            review_text=review_text,
            branch=branch,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        print(f"\nNormalized branch: {branch}")
        print(f"Predicted rating: {rating} ⭐")
        print("Probabilities:")
        for i, p in enumerate(probs, 1):
            print(f"  {i} star: {p:.4f}")
        print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    main()
