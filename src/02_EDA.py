
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import seaborn as sns

df = pd.read_csv("../data/review_clean.csv")

# Frequency Count of Ratings
# print(df.Rating.value_counts())

# Classify into three sentiment groups
def classify_ratings(x):
    if x <= 2:
        return "neg"
    elif x >= 4:
        return "pos"
    else:
        return "neu"

df["Sentiment"] = df["Rating"].apply(lambda x: classify_ratings(x))
# df.to_csv("data/review_sentiment.csv", index=False)

df = pd.read_csv("../data/review_sentiment.csv")

sentiment_counts = df.Sentiment.value_counts()

# Frequency Counts by Sentiment
sentiment_counts.plot(kind='bar', color='skyblue')
plt.title('Frequency Counts of Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Text Length
df['text_length'] = df['Review_Text_clean'].str.len()

plt.figure(figsize=(8,5))
sns.histplot(df["text_length"], kde=True, bins=50)
plt.title("Distribution of Review Length (Word Count)")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Sentiment", y="text_length")
plt.title("Review Length by Sentiment")
plt.show()


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def get_ngram_df(texts, ngram_range=(2,2), top_n=20, stopwords=None):
    # Remove null or empty strings
    texts = [t for t in texts if isinstance(t, str) and t.strip() != ""]

    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stopwords)
    X = vectorizer.fit_transform(texts)

    # Sum up the counts of each token
    counts = X.sum(axis=0).A1
    ngrams = vectorizer.get_feature_names_out()

    df_ngram = pd.DataFrame({
        "ngram": ngrams,
        "count": counts
    }).sort_values("count", ascending=False).head(top_n)

    return df_ngram.reset_index(drop=True)
texts = df["Review_Text_clean"].astype(str).tolist()

bigram_df = get_ngram_df(texts, ngram_range=(2,2), top_n=20)
trigram_df = get_ngram_df(texts, ngram_range=(3,3), top_n=20)

plt.figure(figsize=(10, 5))
plt.barh(bigram_df["ngram"][::-1], bigram_df["count"][::-1])
plt.title("Top 20 Bigrams")
plt.xlabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.barh(trigram_df["ngram"][::-1], trigram_df["count"][::-1])
plt.title("Top 20 Trigrams")
plt.xlabel("Count")
plt.tight_layout()
plt.show()


df["Review_Text_clean"] = df["Review_Text_clean"].apply(
    lambda x: " ".join([w for w in x.split() if len(w) > 2])
)

custom_stopwords = {"disneyland", "disney", "park"}
stopwords = STOPWORDS.union(custom_stopwords)

# Generate WordCloud for each sentiment
sentiments = ["pos", "neu", "neg"]
for sent in sentiments:

    # Filter reviews for specific sentiment
    reviews = df[df["Sentiment"] == sent]["Review_Text_clean"].astype(str).values

    # Combine text
    combined_pos_reviews = " ".join(reviews)

    # Create wordcloud
    positive_wordcloud = WordCloud(
        max_font_size=50,
        max_words=150,
        background_color="white",
        stopwords=stopwords,
    ).generate(combined_pos_reviews)

    # Plot
    plt.figure(figsize=(10,6))
    plt.title(f"{sent} Tweets - Wordcloud")
    plt.imshow(positive_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()



# Extract top-N most frequent words for each sentiment.
# def top_words(df,sent, n=50):
#     words = " ".join(df[df["Sentiment"] == sent]["Review_Text_clean"]).split()
#     return set([w for w, c in Counter(words).most_common(n)])
#
# pos_top = top_words(df,"pos", 100)
# neu_top = top_words(df,"neu", 100)
# neg_top = top_words(df,"neg", 100)
#
# Generate number of overlapping words in top words for different sentiment
# print("Number of Overlapping Words between Positive & Neutral:", len(pos_top.intersection(neu_top)))
# print("Number of Overlapping Words between Negative & Neutral:", len(neu_top.intersection(neg_top)))
# print("Number of Overlapping Words between Positive & Negative::", len(pos_top.intersection(neg_top)))






