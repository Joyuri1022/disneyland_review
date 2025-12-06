import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score

# Load and cache data
@st.cache_data
def load_data():
    # Load the Disneyland reviews dataset
    df = pd.read_csv("data/review_raw.csv", encoding="latin1")
    # Remove rows where Year_Month is missing or invalid
    df = df[df["Year_Month"].str.match(r"^\d{4}-\d{2}$", na=False)]
    # Add separate Year and Month columns (numeric) from Year_Month for analysis
    df["Year"] = df["Year_Month"].str.split("-").str[0].astype(int)
    df["Month"] = df["Year_Month"].str.split("-").str[1].astype(int)
    # If needed, duplicate review text into 'Review_Text_clean' for the model
    if "Review_Text_clean" not in df.columns:
        df["Review_Text_clean"] = df["Review_Text"]
    # Create a sentiment label column: 0 = negative (rating <=2), 1 = neutral (rating=3), 2 = positive (rating >=4)
    df["Sentiment"] = df["Rating"].apply(lambda r: 0 if r <= 2 else (1 if r == 3 else 2))
    return df

# Load and cache models (TF-IDF logreg and HuggingFace RoBERTa)
@st.cache_resource(allow_output_mutation=True)
def load_models():
    # Load the traditional model and its metrics from joblib file
    logreg_data = joblib.load("models/tfidf_logreg.joblib")
    model_logreg = logreg_data["model"]        # sklearn Pipeline containing TF-IDF + logreg model
    metrics_logreg = logreg_data.get("metrics", None)  # dict of stored metrics (accuracy, f1_macro, etc.)
    # Load the transformer model (RoBERTa) from Hugging Face
    hf_pipeline = pipeline("sentiment-analysis", model="djhua0103/disneyland_roberta")
    return model_logreg, metrics_logreg, hf_pipeline

# Initialize data and models
df = load_data()
model_logreg, metrics_logreg, hf_pipeline = load_models()

# Section 1 – Rating Distribution and Trend Visualization
st.title("Disneyland Reviews Sentiment Dashboard")
st.header("Rating Distribution and Trend Visualization")

# Compute average rating over time (per month) for each Disneyland branch
monthly_avg = df.groupby(["Year_Month", "Branch"])["Rating"].mean().reset_index()
# Convert Year_Month to datetime for proper time-series plotting
monthly_avg["Year_Month"] = pd.to_datetime(monthly_avg["Year_Month"], format="%Y-%m")

# Create a line chart of average rating over time for each branch
fig_line = px.line(
    monthly_avg,
    x="Year_Month", y="Rating", color="Branch",
    title="Average Rating Over Time by Branch",
    labels={"Year_Month": "Year-Month", "Rating": "Average Rating"}
)
fig_line.update_traces(mode="lines+markers")  # show markers on lines for clarity
fig_line.update_layout(xaxis_title="Year-Month", yaxis_title="Average Rating (1-5)")

# Create a histogram of the count distribution of ratings 1–5
fig_hist = px.histogram(
    df, x="Rating", nbins=5,
    title="Distribution of Review Ratings",
    category_orders={"Rating": [1, 2, 3, 4, 5]}
)
fig_hist.update_xaxes(dtick=1)  # ensure each rating (1 through 5) shows on x-axis
fig_hist.update_layout(xaxis_title="Rating (1 = worst, 5 = best)", yaxis_title="Number of Reviews")

# Display the charts
st.plotly_chart(fig_line, use_container_width=True)
st.plotly_chart(fig_hist, use_container_width=True)

# Section 2 – Single Review Sentiment Prediction
st.header("Single Review Sentiment Prediction")
st.write("Enter a review and select its context (park branch, location, date) to predict sentiment using both models:")

# Input widgets for review text and metadata
review_text = st.text_area("Review Text", "")
branch_options = df["Branch"].unique().tolist()
location_options = df["Reviewer_Location"].unique().tolist()
year_month_options = df["Year_Month"].unique().tolist()
branch_input = st.selectbox("Branch", sorted(branch_options))
location_input = st.selectbox("Reviewer Location", sorted(location_options))
date_input = st.selectbox("Review Date (Year-Month)", sorted(year_month_options))

# When the user clicks the Predict button, run sentiment prediction
if st.button("Predict Sentiment"):
    # Prepare a one-sample DataFrame for the logreg model input
    year_str, month_str = str(date_input).split("-")
    sample_df = pd.DataFrame({
        # Use the same column names the model was trained on:
        "Review_Text_clean": [review_text],
        "Branch": [branch_input],
        "Reviewer_Location": [location_input],
        "Year": [int(year_str)],
        "Month": [int(month_str)]
    })
    # If the model expects 'Review_Text' instead of 'Review_Text_clean', include it as well
    if "Review_Text" in df.columns:
        sample_df["Review_Text"] = review_text

    # Get prediction from the TF-IDF + logreg model
    logreg_pred = model_logreg.predict(sample_df)[0]
    # Determine predicted label and confidence for logreg model
    if isinstance(logreg_pred, str):
        logreg_label = logreg_pred  # model returns label as string (e.g. "Positive")
    else:
        # If model returns numeric class (e.g. 0/1/2), map it to sentiment label
        logreg_label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        logreg_label = logreg_label_map.get(logreg_pred, str(logreg_pred))
    # Check if probability is available (e.g. if classifier has predict_proba)
    logreg_conf = None
    if hasattr(model_logreg, "predict_proba"):
        try:
            logreg_prob = model_logreg.predict_proba(sample_df)
            logreg_conf = float(logreg_prob.max())  # highest probability among classes
        except Exception:
            logreg_conf = None

    # Get prediction from the RoBERTa transformer model
    hf_result = hf_pipeline(review_text)
    hf_label_raw = hf_result[0]["label"]   # e.g. "LABEL_2" or "Positive"
    hf_score = hf_result[0]["score"]      # confidence score for the predicted label
    # Normalize the transformer label to Negative/Neutral/Positive
    hf_label = hf_label_raw
    if hf_label_raw.upper().startswith("LABEL"):  # if format like "LABEL_0"
        try:
            label_idx = int(hf_label_raw.split("_")[1])
        except:
            label_idx = None
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        hf_label = label_map.get(label_idx, hf_label_raw)
    else:
        # Use the raw label (or title-case it if it's already a word like "NEGATIVE")
        hf_label = hf_label_raw.capitalize() if hf_label_raw.islower() else hf_label_raw
    # Display the predictions from both models side by side
    col1, col2 = st.columns(2)
    col1.write(f"**TF-IDF Model Prediction:** {logreg_label}" + (f" ({logreg_conf*100:.1f}%)" if logreg_conf is not None else ""))
    col2.write(f"**RoBERTa Model Prediction:** {hf_label} ({hf_score*100:.1f}%)")

# Section 3 – Model Comparison Panel
st.header("Model Comparison")
st.write("Performance of the traditional TF-IDF logreg model vs. the RoBERTa transformer model:")

# Obtain accuracy and F1-macro for each model
# For the logreg model, use stored metrics if available
logreg_accuracy = None
logreg_f1 = None
if metrics_logreg is not None:
    logreg_accuracy = metrics_logreg.get("accuracy", None)
    logreg_f1 = metrics_logreg.get("f1_macro", None)
# If not available (just in case), compute metrics on the entire dataset
if logreg_accuracy is None or logreg_f1 is None:
    true_labels = df["Sentiment"]
    logreg_preds = model_logreg.predict(df[["Review_Text_clean", "Branch", "Reviewer_Location", "Year", "Month"]])
    # If predictions are strings, map them to numeric for scoring
    if isinstance(logreg_preds[0], str):
        label_to_num = {"Negative": 0, "negative": 0, "Neutral": 1, "neutral": 1, "Positive": 2, "positive": 2}
        logreg_preds_num = [label_to_num.get(x, x) for x in logreg_preds]
    else:
        logreg_preds_num = logreg_preds
    logreg_accuracy = accuracy_score(true_labels, logreg_preds_num)
    logreg_f1 = f1_score(true_labels, logreg_preds_num, average="macro")

# For the RoBERTa model, compute metrics on a smaller test subset of the data
# We take a stratified sample to include examples of each sentiment class
neg_samples = df[df["Sentiment"] == 0].sample(n=min(50, len(df[df["Sentiment"] == 0])), random_state=39)
neu_samples = df[df["Sentiment"] == 1].sample(n=min(50, len(df[df["Sentiment"] == 1])), random_state=39)
pos_samples = df[df["Sentiment"] == 2].sample(n=min(50, len(df[df["Sentiment"] == 2])), random_state=39)
test_df = pd.concat([neg_samples, neu_samples, pos_samples])
test_true = test_df["Sentiment"].tolist()

# Predict with RoBERTa on each sample in the subset
hf_preds = []
for text in test_df["Review_Text"]:
    MAX_LEN = 256

    res = hf_pipeline(
        text,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )
    raw_label = res[0]["label"]
    # Map raw label to numeric as before
    if raw_label.upper().startswith("LABEL"):
        idx = int(raw_label.split("_")[1])
    else:
        raw_label_low = raw_label.lower()
        if raw_label_low == "negative":
            idx = 0
        elif raw_label_low == "neutral":
            idx = 1
        elif raw_label_low == "positive":
            idx = 2
        else:
            idx = None
    hf_preds.append(idx)
# Compute accuracy and F1 (macro) for RoBERTa on this test subset
hf_accuracy = accuracy_score(test_true, hf_preds)
hf_f1 = f1_score(test_true, hf_preds, average="macro")

# Create a summary table of the performance metrics
metrics_df = pd.DataFrame({
    "Model": ["TF-IDF + logreg", "RoBERTa (Transformer)"],
    "Accuracy": [logreg_accuracy, hf_accuracy],
    "F1-macro": [logreg_f1, hf_f1]
})
# Round the metrics for display
metrics_df["Accuracy"] = metrics_df["Accuracy"].apply(lambda x: round(x, 3) if x is not None else None)
metrics_df["F1-macro"] = metrics_df["F1-macro"].apply(lambda x: round(x, 3) if x is not None else None)

# Display the table of metrics
st.table(metrics_df.set_index("Model"))

# Section 4 – TF-IDF Interpretability
st.header("TF-IDF Model Interpretability")
st.write("Top words contributing to each sentiment (as identified by the TF-IDF + logreg model):")

# Extract the TF-IDF vectorizer and the classifier from the pipeline
# (Assuming the pipeline step names as in training: 'preprocess' -> ColumnTransformer, 'clf' -> classifier)
try:
    preprocessor = model_logreg.named_steps["preprocess"]
    text_vectorizer = preprocessor.named_transformers_["text"]
except Exception as e:
    text_vectorizer = None

top_words = {"Negative": [], "Neutral": [], "Positive": []}
if text_vectorizer is not None:
    # Get feature names for text features (TF-IDF vocabulary)
    feature_names = text_vectorizer.get_feature_names_out()
    num_text_features = len(feature_names)
    # Get the classifier coefficients (for LinearSVC or LogisticRegression)
    classifier = model_logreg.named_steps.get("clf", None)
    if classifier is None:
        classifier = model_logreg  # model might itself be a classifier if no pipeline
    if hasattr(classifier, "coef_"):
        coefs = classifier.coef_
        classes = classifier.classes_
        # Determine class ordering and names
        class_names = []
        if isinstance(classes[0], str):
            # If classes are strings (e.g., "Negative"), use them (capitalized)
            class_names = [cls.capitalize() if cls.islower() else cls for cls in classes]
        else:
            # Numeric classes assumed 0,1,2 mapping to Negative, Neutral, Positive
            class_names = ["Negative", "Neutral", "Positive"]
        # For each class, find top weighted words
        for idx, class_name in enumerate(class_names):
            if idx < coefs.shape[0]:
                # Coefficients for this class (one-vs-rest for LinearSVC or multi-class for LogisticRegression)
                coef_values = coefs[idx][:num_text_features]
                # Get indices of top 10 highest coefficients
                top_idx = coef_values.argsort()[::-1][:10]
                top_words[class_name] = [feature_names[j] for j in top_idx]
    else:
        st.write("The logreg model does not have linear coefficients for interpretability.")
else:
    st.write("Text vectorizer not found in the model pipeline.")

# Display the top words for each sentiment side by side
neg_words = top_words.get("Negative", [])
neu_words = top_words.get("Neutral", [])
pos_words = top_words.get("Positive", [])
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Negative")
    for w in neg_words:
        st.write(f"- {w}")
with col2:
    st.subheader("Neutral")
    for w in neu_words:
        st.write(f"- {w}")
with col3:
    st.subheader("Positive")
    for w in pos_words:
        st.write(f"- {w}")
