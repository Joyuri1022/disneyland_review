🎢 Disneyland Review Sentiment Analysis

A Comparative Study of Classical ML, Sentence Embeddings, and Pretrained Transformers


📌 Overview

This project performs sentiment classification on Disneyland customer reviews using three major modeling approaches:

Classical Machine Learning (TF-IDF + Naive Bayes / SVM)

Sentence Embedding Models (SBERT + LightGBM / classifiers)

Pretrained Transformers (DistilBERT, BERT, RoBERTa, DeBERTa)

The goal is to evaluate how different NLP methods perform on imbalanced sentiment data and identify which approach generalizes best across positive, neutral, and negative classes.

A Streamlit application (app.py) is also included for interactive visualization and model inference.


## 📂 Project Structure

```text
code/
│
├── data/                       # Raw and cleaned datasets
├── models/                     # Local model artifacts (only contain classical ML)
│
├── src/                        # All training, preprocessing, and analysis scripts
│   ├── 02_EDA.py               # Exploratory Data Analysis
│   ├── preprocess.py           # Data preprocessing pipeline
│   │
│   ├── train_nb_svm.py         # Classical ML: TF-IDF + NB/SVM
│   ├── train_tfidf.py          # Classical ML: TF-IDF + LogReg
│   │
│   ├── train_sbert_embeddings.py   # SBERT embedding generation
│   ├── train_sbert_lightgbm.py     # SBERT + LightGBM classifier
│   │
│   ├── train_distillbert.py    # basic DistilBERT 
│   ├── train_distillbert_v2.py # DistilBERT with class weight
│   ├── train_distillbert_v3.py # DistilBERT with Sentiment (class -> sentiment)
│   │
│   ├── train_bert.py           # BERT fine-tuning
│   ├── train_roberta_final.py  # Final RoBERTa fine-tuning (weighted)
│   ├── train_roberta_metrics.py# Evaluation script for RoBERTa model
│   │
│   ├── train_deberta_base.py   # base DeBERTa 
│   ├── train_deberta_fl.py     # small DeBERTa + focal loss experiment
│   │
│   ├── predict_tfidf.py        # TF-IDF prediction script
│   └── predict_distillbert.py  # DistilBERT prediction script
│
├── app.py                      #  Experimental/alternative Streamlit application
└── app2.py                     #  Our Final Streamlit version
```

🔐 Model Storage Strategy

Classical ML models → stored locally under code/models/

SBERT-based and Transformer models  (DistilBERT, RoBERTa, DeBERTa, BERT) → stored on Hugging Face Hub

Only metrics/checkpoints needed by Streamlit are saved locally


📊 Dataset

The dataset contains Disneyland visitor reviews with the following fields:

Review_Text

Rating (1–5)

Branch (California / Paris / Hong Kong)

Year_Month

Sentiment labels are mapped into:

positive

neutral

negative


🔍 Exploratory Data Analysis

Run 02_EDA.py to generate:

Rating distribution by branch

Monthly rating trends

Word clouds & n-gram frequencies

Sentiment distribution

Missing-value checks & preprocessing decisions

Run branch.py & distribution.py in data to generate:

Average rating over time by branch

Reviews count by branch



🧹 Preprocessing Pipeline

Implemented in preprocess.py, including:

Text cleaning & normalization

Tokenization & lemmatization

Removing corrupt timestamps

Splitting train/validation/test sets

Handling class imbalance (class weights / weighted sampler)


⚙️ Model Training Scripts
1. Classical Machine Learning

Scripts:

train_nb_svm.py

train_tfidf.py

Models:

TF-IDF + Multinomial Naive Bayes

TF-IDF + Linear SVM

Saved under: code/models/

2. Sentence Embedding Models

Scripts:

train_sbert_embeddings.py

train_sbert_lightgbm.py

Models:

SBERT embeddings + LightGBM

SBERT embeddings + Logistic Regression/SVM

Stored locally.

3. Pretrained Transformers (Hugging Face)
DistilBERT

train_distillbert.py

train_distillbert_v2.py

train_distillbert_v3.py (weighted loss)

BERT

train_bert.py

RoBERTa

train_roberta_final.py

train_roberta_metrics.py

DeBERTa

train_deberta_base.py

train_deberta_fl.py (focal loss)

All trained models are uploaded to Hugging Face and loaded during inference.


🏆 Results Summary

All models perform strongly on the positive class.

Performance drops on neutral and negative, especially neutral recall.

Pretrained transformers deliver the best performance, robust across all metrics.

Sentence embedding models rank second.

Classical ML models perform the weakest due to limited expressive power.


💡 Streamlit Demo

Run the application:

pip install -r requirements.txt
streamlit run code/app2.py


The app includes:

EDA visualizations

Model comparisons

Real-time sentiment prediction with Hugging Face models
