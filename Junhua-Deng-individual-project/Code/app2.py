import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import json
from transformers import pipeline

# ---------- Page config ----------
st.set_page_config(page_title="Disneyland Sentiment Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #050816 !important;  
        color: #f9fafb !important;            
    }
    html, body, [data-testid="stMarkdownContainer"],
    .stMarkdown, .stText, .stExpander, .stTable, .stDataFrame, .stMetric {
        color: #f9fafb !important;
    }
    table, th, td {
        color: #f9fafb !important;
    }
    button[aria-expanded="true"], button[aria-expanded="false"] {
        color: #f9fafb !important;
    }
    button[kind="primary"] {
        background-color: #4f46e5 !important;
        color: #f9fafb !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DARK_BG = "#050816"
FONT_COLOR = "#f9fafb"


# ---------- Load & cache data ----------
@st.cache_data
def load_data():
    # Load the Disneyland reviews dataset
    df = pd.read_csv("data/review_raw.csv", encoding="latin1")

    # Remove rows where Year_Month is missing or invalid (not in YYYY-MM format)
    df = df[df["Year_Month"].str.match(r"^\d{4}-\d{2}$", na=False)]

    # Add separate Year and Month columns (numeric) from Year_Month for analysis
    df["Year"] = df["Year_Month"].str.split("-").str[0].astype(int)
    df["Month"] = df["Year_Month"].str.split("-").str[1].astype(int)

    # If needed, duplicate review text into 'Review_Text_clean' for the model
    if "Review_Text_clean" not in df.columns:
        df["Review_Text_clean"] = df["Review_Text"]

    # Create a sentiment label column:
    # 0 = negative (rating <=2), 1 = neutral (rating=3), 2 = positive (rating >=4)
    df["Sentiment"] = df["Rating"].apply(
        lambda r: 0 if r <= 2 else (1 if r == 3 else 2)
    )
    return df


# ---------- Load & cache models ----------
@st.cache_resource
def load_models():
    # Load the traditional model and its metrics from joblib file
    logreg_data = joblib.load("models/tfidf_logreg.joblib")

    # New format: {"model": ..., "metrics": {...}}
    if isinstance(logreg_data, dict) and "model" in logreg_data:
        model_logreg = logreg_data["model"]
        metrics_logreg = logreg_data.get("metrics", None)
    else:
        # Fallback: just a pipeline without metrics
        model_logreg = logreg_data
        metrics_logreg = None

    # Load the transformer model (RoBERTa) from Hugging Face
    hf_pipe = pipeline("sentiment-analysis", model="djhua0103/disneyland_roberta")

    return model_logreg, metrics_logreg, hf_pipe


# ---------- Initialize data and models ----------
df = load_data()
model_logreg, metrics_logreg, hf_pipeline = load_models()

st.title("Disneyland Reviews Sentiment Dashboard")

# Top-level tabs for different modules
tab_overview, tab_single, tab_compare, tab_tfidf = st.tabs(
    ["Overview", "Single Prediction", "Model Comparison", "TF-IDF Interpretability"]
)

# =====================================================================
# TAB 1 – Overview: Rating Distribution and Trend Visualization
# =====================================================================
with tab_overview:
    st.header("Rating Distribution and Trend Visualization")

    # Compute average rating over time (per month) for each Disneyland branch
    monthly_avg = df.groupby(["Year_Month", "Branch"])["Rating"].mean().reset_index()
    # Convert Year_Month to datetime for proper time-series plotting
    monthly_avg["Year_Month"] = pd.to_datetime(monthly_avg["Year_Month"], format="%Y-%m")

    # Layout: left = time series, right = histogram
    col_left, col_right = st.columns(2)

    # Line chart: average rating over time by branch
    with col_left:
        fig_line = px.line(
            monthly_avg,
            x="Year_Month",
            y="Rating",
            color="Branch",
            title="Average Rating Over Time by Branch",
            labels={"Year_Month": "Year-Month", "Rating": "Average Rating"},
        )
        fig_line.update_traces(mode="lines+markers")

        fig_line.update_layout(
            template="plotly_dark",
            xaxis_title="Year-Month",
            yaxis_title="Average Rating (1-5)",
            legend_title="Branch",
            plot_bgcolor=DARK_BG,
            paper_bgcolor=DARK_BG,
            font=dict(color=FONT_COLOR),

            title=dict(font=dict(color=FONT_COLOR)),
            legend=dict(
                title=dict(font=dict(color=FONT_COLOR)),
                font=dict(color=FONT_COLOR),
            ),
            xaxis=dict(title=dict(font=dict(color=FONT_COLOR))),
            yaxis=dict(title=dict(font=dict(color=FONT_COLOR))),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # Histogram: rating distribution 1–5
    with col_right:
        fig_hist = px.histogram(
            df,
            x="Rating",
            nbins=5,
            title="Distribution of Review Ratings",
            category_orders={"Rating": [1, 2, 3, 4, 5]},
        )
        fig_hist.update_xaxes(dtick=1)
        fig_hist.update_layout(
            template="plotly_dark",
            xaxis_title="Rating (1 = worst, 5 = best)",
            yaxis_title="Number of Reviews",
            plot_bgcolor=DARK_BG,
            paper_bgcolor=DARK_BG,
            font=dict(color=FONT_COLOR),

            title=dict(font=dict(color=FONT_COLOR)),
            legend=dict(
                title=dict(font=dict(color=FONT_COLOR)),
                font=dict(color=FONT_COLOR),
            ),
            xaxis=dict(title=dict(font=dict(color=FONT_COLOR))),
            yaxis=dict(title=dict(font=dict(color=FONT_COLOR))),
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# =====================================================================
# TAB 2 – Single Review Sentiment Prediction
# =====================================================================
with tab_single:
    st.header("Single Review Sentiment Prediction")
    st.write(
        "Enter a review and select its context (park branch, location, date) "
        "to predict sentiment using both models:"
    )

    # Input widgets for review text and metadata
    review_text = st.text_area("Review Text", "")

    branch_options = sorted(df["Branch"].unique().tolist())
    location_options = sorted(df["Reviewer_Location"].unique().tolist())
    year_month_options = sorted(df["Year_Month"].unique().tolist())

    col_b, col_l, col_d = st.columns(3)
    branch_input = col_b.selectbox("Branch", branch_options)
    location_input = col_l.selectbox("Reviewer Location", location_options)
    date_input = col_d.selectbox("Review Date (Year-Month)", year_month_options)

    if st.button("Predict Sentiment", type="primary"):
        if not review_text.strip():
            st.warning("Please enter a review text first.")
        else:
            # Prepare a one-sample DataFrame for the logreg model input
            year_str, month_str = str(date_input).split("-")
            sample_df = pd.DataFrame(
                {
                    "Review_Text_clean": [review_text],
                    "Branch": [branch_input],
                    "Reviewer_Location": [location_input],
                    "Year": [int(year_str)],
                    "Month": [int(month_str)],
                }
            )
            if "Review_Text" in df.columns:
                sample_df["Review_Text"] = review_text

            # --- TF-IDF + Logistic Regression prediction ---
            logreg_pred = model_logreg.predict(sample_df)[0]
            if isinstance(logreg_pred, str):
                logreg_label = logreg_pred
            else:
                logreg_label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                logreg_label = logreg_label_map.get(logreg_pred, str(logreg_pred))

            logreg_conf = None
            if hasattr(model_logreg, "predict_proba"):
                try:
                    logreg_prob = model_logreg.predict_proba(sample_df)
                    logreg_conf = float(logreg_prob.max())
                except Exception:
                    logreg_conf = None

            # --- RoBERTa transformer prediction ---
            hf_result = hf_pipeline(review_text)
            hf_label_raw = hf_result[0]["label"]
            hf_score = hf_result[0]["score"]

            if hf_label_raw.upper().startswith("LABEL"):
                try:
                    label_idx = int(hf_label_raw.split("_")[1])
                except Exception:
                    label_idx = None
                label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                hf_label = label_map.get(label_idx, hf_label_raw)
            else:
                # Normal labels like "negative", "positive", etc.
                raw_low = hf_label_raw.lower()
                if raw_low == "negative":
                    hf_label = "Negative"
                elif raw_low == "neutral":
                    hf_label = "Neutral"
                elif raw_low == "positive":
                    hf_label = "Positive"
                else:
                    hf_label = hf_label_raw

            # Show predictions side by side
            c1, c2 = st.columns(2)
            with c1:
                if logreg_conf is not None:
                    st.markdown(
                        f"**TF-IDF Model Prediction:** {logreg_label} "
                        f"({logreg_conf * 100:.1f}%)"
                    )
                else:
                    st.markdown(f"**TF-IDF Model Prediction:** {logreg_label}")
            with c2:
                st.markdown(
                    f"**RoBERTa Model Prediction:** {hf_label} "
                    f"({hf_score * 100:.1f}%)"
                )


# =====================================================================
# TAB 3 – Model Comparison (metrics + classification reports, from files)
# =====================================================================
with tab_compare:
    st.header("Model Comparison")
    st.write(
        "Performance of the traditional TF-IDF logistic regression model "
        "versus the RoBERTa transformer model (loaded from saved metrics)."
    )

    # ---------- TF-IDF + LogReg metrics from joblib ----------
    logreg_accuracy = None
    logreg_f1 = None
    logreg_cls_report = None

    if metrics_logreg is not None:
        logreg_accuracy = metrics_logreg.get("accuracy", None)
        logreg_f1 = metrics_logreg.get("f1_macro", None)
        logreg_cls_report = metrics_logreg.get("classification_report", None)

    # ---------- RoBERTa metrics from JSON ----------
    def load_roberta_metrics():
        candidate_paths = [
            "models/metrics_roberta.json",
            "models/roberta-base_weighted/metrics_roberta.json",
        ]
        for path in candidate_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                continue
        return {}

    metrics_roberta = load_roberta_metrics()

    hf_accuracy = metrics_roberta.get("eval_accuracy", None)
    hf_f1 = metrics_roberta.get("eval_f1_macro", None)
    hf_cls_report = metrics_roberta.get("classification_report", None)

    # ---------- Summary metrics table ----------
    metrics_df = pd.DataFrame(
        {
            "Model": ["TF-IDF + LogReg", "RoBERTa (Transformer)"],
            "Accuracy": [logreg_accuracy, hf_accuracy],
            "F1-macro": [logreg_f1, hf_f1],
        }
    )

    metrics_df["Accuracy"] = metrics_df["Accuracy"].apply(
        lambda x: round(x, 3) if isinstance(x, (int, float)) else x
    )
    metrics_df["F1-macro"] = metrics_df["F1-macro"].apply(
        lambda x: round(x, 3) if isinstance(x, (int, float)) else x
    )

    st.subheader("Summary Metrics")
    st.table(metrics_df.set_index("Model"))

    # ---------- Classification reports ----------
    st.subheader("Classification Reports (from saved metrics)")

    c_left, c_right = st.columns(2)
    with c_left:
        with st.expander("TF-IDF + LogReg Classification Report", expanded=False):
            if logreg_cls_report:
                st.text(logreg_cls_report)
            else:
                st.info("No classification_report found in TF-IDF metrics.")

    with c_right:
        with st.expander("RoBERTa Classification Report", expanded=False):
            if hf_cls_report:
                st.text(hf_cls_report)
            else:
                st.info("No classification_report found in RoBERTa metrics JSON.")


# =====================================================================
# TAB 4 – TF-IDF Model Interpretability
# =====================================================================
with tab_tfidf:
    st.header("TF-IDF Model Interpretability")
    st.write(
        "Top words contributing to each sentiment "
        "(as identified by the TF-IDF + logistic regression model)."
    )

    # Extract the TF-IDF vectorizer and the classifier from the pipeline
    try:
        preprocessor = model_logreg.named_steps["preprocess"]
        text_vectorizer = preprocessor.named_transformers_["text"]
    except Exception:
        text_vectorizer = None

    top_words = {"Negative": [], "Neutral": [], "Positive": []}

    if text_vectorizer is not None:
        feature_names = text_vectorizer.get_feature_names_out()
        num_text_features = len(feature_names)

        classifier = model_logreg.named_steps.get("clf", None)
        if classifier is None:
            classifier = model_logreg

        if hasattr(classifier, "coef_"):
            coefs = classifier.coef_
            classes = classifier.classes_

            # Determine class name ordering
            if isinstance(classes[0], str):
                class_names = [
                    cls.capitalize() if cls.islower() else cls for cls in classes
                ]
            else:
                class_names = ["Negative", "Neutral", "Positive"]

            for idx, class_name in enumerate(class_names):
                if idx < coefs.shape[0]:
                    coef_values = coefs[idx][:num_text_features]
                    top_idx = coef_values.argsort()[::-1][:10]
                    top_words[class_name] = [feature_names[j] for j in top_idx]
        else:
            st.info(
                "The logistic regression classifier does not expose linear "
                "coefficients for interpretability."
            )
    else:
        st.info("Text vectorizer not found in the model pipeline.")

    neg_words = top_words.get("Negative", [])
    neu_words = top_words.get("Neutral", [])
    pos_words = top_words.get("Positive", [])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Negative")
        if neg_words:
            for w in neg_words:
                st.write(f"- {w}")
        else:
            st.write("No keywords available.")
    with c2:
        st.subheader("Neutral")
        if neu_words:
            for w in neu_words:
                st.write(f"- {w}")
        else:
            st.write("No keywords available.")
    with c3:
        st.subheader("Positive")
        if pos_words:
            for w in pos_words:
                st.write(f"- {w}")
        else:
            st.write("No keywords available.")
