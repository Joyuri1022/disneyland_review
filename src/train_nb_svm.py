#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a TF-IDF + Logistic Regression baseline model
for Disneyland review rating prediction (1–5 stars).

Input: a preprocessed CSV created by preprocess.py

Output:
  - Prints evaluation metrics on a validation set
  - Saves a trained sklearn Pipeline (preprocessing + classifier)
"""

import os
import argparse

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def load_data(csv_path: str):
    """
    Load the preprocessed dataset from CSV.
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Converts sentiment into numerical labels
    sentiment_map = {"neg": 0, "neu": 1, "pos": 2}
    df["label"] = df["Sentiment"].map(sentiment_map)

    texts = df["Review_Text_clean"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def build_pipeline(model_type: str = "logreg") -> Pipeline:
    text_transformer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    model_type = model_type.lower()

    if model_type == "logreg":
        clf = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=-1
        )
    elif model_type == "nb":
        clf = MultinomialNB()
    elif model_type == "svm":
        clf = LinearSVC(class_weight="balanced")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipe = Pipeline(
        steps=[
            ("tfidf", text_transformer),
            ("clf", clf),
        ]
    )

    return pipe



def main():

    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Naive Bayes model for Disneyland ratings."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the preprocessed CSV file.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["logreg", "nb", "svm"],
        default="logreg",
        help="Classifier type: 'logreg', 'nb', or 'svm' (default: logreg).",
    )

    parser.add_argument(
        "--model_out",
        type=str,
        default="../models/tfidf_logreg.joblib",
        help="Path to save the trained model pipeline.",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Validation split size (default: 0.2).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=39,
        help="Random seed for train/validation split.",
    )

    args = parser.parse_args()


    X, y = load_data(args.input)

    # Train/validation split (stratified by rating)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    model_type = args.model_type

    # Build pipeline
    model = build_pipeline(model_type=model_type)

    # Fit
    print(f"Training {model_type} model (TF-IDF)...")
    if model_type.lower() == "nb":
        # First compute sample weights based on class distribution
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

        model.fit(X_train, y_train, clf__sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    # Evaluate
    # print("Evaluating on validation set...")
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")

    print(f"\nValidation Accuracy: {acc:.4f}")
    print(f"Validation F1-macro: {f1_macro:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    model_out = args.model_out

    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    # Save model
    print(f"\nSaving trained model to: {model_out}")
    joblib.dump(model, model_out)
    print("Model saved.")


if __name__ == "__main__":
    main()
