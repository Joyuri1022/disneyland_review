#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a TF-IDF + Logistic Regression baseline model
for Disneyland review sentiment prediction (negative / neutral / positive).

Original 1–5 star ratings are mapped to:
    1–2 -> negative
    3   -> neutral
    4–5 -> positive

Input: a preprocessed CSV created by preprocess.py
       (must contain columns:
        Rating, Review_Text_clean, Branch, Reviewer_Location, Year, Month)

Output:
  - Prints evaluation metrics on a validation set
  - Saves a trained sklearn Pipeline (preprocessing + classifier)
    together with evaluation metrics in a joblib file:
        {"model": pipeline, "metrics": {...}}
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the preprocessed dataset from CSV.
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = [
        "Rating",
        "Review_Text_clean",
        "Branch",
        "Reviewer_Location",
        "Year",
        "Month",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in CSV.")

    # Drop rows with missing critical fields
    before = len(df)
    df = df.dropna(subset=["Rating", "Review_Text_clean"])
    print(f"Dropped {before - len(df)} rows with missing Rating/Text.")

    return df


def map_rating_to_sentiment(r: int) -> str:
    """
    Map original 1–5 rating to sentiment label:
        1–2 -> "negative"
        3   -> "neutral"
        4–5 -> "positive"
    """
    if r <= 2:
        return "negative"
    elif r == 3:
        return "neutral"
    else:
        return "positive"


def build_pipeline() -> Pipeline:
    """
    Build a sklearn Pipeline that includes:
    - ColumnTransformer:
        * TF-IDF on 'Review_Text_clean'
        * One-hot encoding on 'Branch' and 'Reviewer_Location'
        * Pass through numeric 'Year' and 'Month'
    - LogisticRegression classifier
    """
    text_col = "Review_Text_clean"
    cat_cols = ["Branch", "Reviewer_Location"]
    num_cols = ["Year", "Month"]

    # Preprocess text column with TF-IDF
    text_transformer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    # Preprocess categorical columns with one-hot encoding
    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, text_col),
            ("cat", cat_transformer, cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    # Logistic Regression classifier (multiclass over 3 sentiment labels)
    clf = LogisticRegression(
        max_iter=2000,

        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1
    )

    # Full pipeline
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    return pipe


def train_and_evaluate(
    df: pd.DataFrame,
    model_out: str,
    test_size: float = 0.2,
    random_state: int = 39
):
    """
    Split the data, train the pipeline, evaluate on validation set,
    and save the trained model together with metrics.
    """
    # Features
    X = df[
        [
            "Review_Text_clean",
            "Branch",
            "Reviewer_Location",
            "Year",
            "Month",
        ]
    ]

    # Map original 1–5 ratings to sentiment labels
    ratings = df["Rating"].astype(int)
    y = ratings.apply(map_rating_to_sentiment)

    # Train/validation split (stratified by sentiment label)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    # Build pipeline
    model = build_pipeline()

    # Fit
    print("Training Logistic Regression model (TF-IDF + metadata, 3-class sentiment)...")
    model.fit(X_train, y_train)

    # Evaluate
    print("Evaluating on validation set...")
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")

    print(f"\nValidation Accuracy: {acc:.4f}")
    print(f"Validation F1-macro: {f1_macro:.4f}\n")

    print("Classification Report:")
    report = classification_report(y_val, y_pred, digits=4)
    print(report)

    print("Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred, labels=np.unique(y_val))
    print(cm)

    # Prepare metrics dict to be saved with the model
    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": list(np.unique(y_val)),
    }

    # Ensure output directory exists
    out_dir = os.path.dirname(model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save model + metrics
    print(f"\nSaving trained model and metrics to: {model_out}")
    joblib.dump(
        {
            "model": model,
            "metrics": metrics,
        },
        model_out,
    )
    print("Model and metrics saved.")


def main():

    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Logistic Regression model for Disneyland sentiment (neg/neu/pos)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the preprocessed CSV file.",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="../models/tfidf_logreg.joblib",
        help="Path to save the trained model pipeline and metrics.",
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

    df = load_data(args.input)
    df["Year"] = df["Year"].fillna(df["Year"].mode()[0])
    df["Month"] = df["Month"].fillna(df["Month"].mode()[0])

    train_and_evaluate(
        df,
        model_out=args.model_out,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
