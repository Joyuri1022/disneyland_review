#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocess the Disneyland Reviews dataset.

This script performs:
1. Reading the raw CSV file (required columns:
   Review_ID, Rating, Year_Month, Reviewer_Location, Review_Text, Branch)
2. Cleaning the review text:
   - Lowercasing
   - Removing URLs / HTML tags
   - Removing punctuation and numbers
   - Tokenization
   - Stopword removal
   - Lemmatization
3. Parsing the Year_Month field into separate Year and Month columns.
4. Creating three different text fields:
   - Review_Text_raw   : original text
   - Review_Text_clean : cleaned text for TF-IDF & classical ML models
   - Review_Text_bert  : branch + raw text combined, for BERT fine-tuning
5. Saving the processed dataset to a new CSV file.
"""

import re
import argparse
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_resources():
    """
    Ensure that required NLTK resources are downloaded.
    Downloads punkt, stopwords, and wordnet if missing.
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {name} ...")
            nltk.download(name)


def clean_text_basic(text: str) -> str:
    """
    Perform basic cleaning on the input text:
    - Lowercase
    - Remove URLs
    - Remove HTML tags
    - Keep only alphabetic characters
    - Normalize multiple spaces
    """
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text: str,
                    stop_words_set,
                    lemmatizer: WordNetLemmatizer) -> str:
    """
    Full text preprocessing pipeline:
    1. Basic cleaning
    2. Tokenization
    3. Stopword removal
    4. Lemmatization

    Returns a cleaned string with tokens separated by spaces.
    """
    text = clean_text_basic(text)
    if not text:
        return ""

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords & very short tokens
    tokens = [t for t in tokens if t not in stop_words_set and len(t) > 1]

    # lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def parse_year_month(col: pd.Series):
    """
    Parse the 'Year_Month' column (e.g., '2019/5/1') into:
    - Year
    - Month
    """
    dt = pd.to_datetime(col, errors="coerce")
    year = dt.dt.year
    month = dt.dt.month
    return year, month


def main(input_path: str, output_path: str):
    print(f"Reading data from: {input_path}")
    df = pd.read_csv(input_path, encoding="latin1")

    # Check required columns
    required_cols = [
        "Review_ID",
        "Rating",
        "Year_Month",
        "Reviewer_Location",
        "Review_Text",
        "Branch",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in input CSV.")

    # Drop rows with missing Rating or Review_Text
    before = len(df)
    df = df.dropna(subset=["Rating", "Review_Text"])
    print(f"Dropped {before - len(df)} rows with missing Rating or Review_Text.")

    # Keep only valid ratings (1–5)
    df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5)]
    print(f"After filtering valid ratings: {len(df)} rows remain.")

    # Parse Year_Month → Year, Month
    df["Year"], df["Month"] = parse_year_month(df["Year_Month"])

    # Save raw text
    df["Review_Text_raw"] = df["Review_Text"].astype(str)

    # Ensure NLTK resources exist
    ensure_nltk_resources()
    stop_words_set = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Clean text for TF-IDF models
    print("Cleaning text (may take a while)...")
    df["Review_Text_clean"] = df["Review_Text_raw"].apply(
        lambda x: preprocess_text(x, stop_words_set, lemmatizer)
    )

    # Build text for BERT models
    # Example:
    #   "Branch: Tokyo. Review: The parade was amazing..."
    df["Review_Text_bert"] = df.apply(
        lambda row: f"Branch: {row['Branch']}. Review: {row['Review_Text_raw']}",
        axis=1,
    )

    # Select output columns
    keep_cols = [
        "Review_ID",
        "Rating",
        "Year",
        "Month",
        "Reviewer_Location",
        "Branch",
        "Review_Text_raw",
        "Review_Text_clean",
        "Review_Text_bert",
    ]
    df_out = df[keep_cols]

    # Save processed file
    print(f"Saving cleaned data to: {output_path}")
    df_out.to_csv(output_path, index=False)
    print("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Disneyland Reviews dataset")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the raw CSV file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the processed CSV file.")
    args = parser.parse_args()

    main(args.input, args.output)
