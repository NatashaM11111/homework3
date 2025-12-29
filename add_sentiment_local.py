import pandas as pd
from transformers import pipeline

# 1) Load your scraped reviews
df = pd.read_csv("reviews.csv")

# 2) Make sure you have a text column called "text"
if "text" not in df.columns:
    raise ValueError("Expected a 'text' column in reviews.csv")

df["text"] = df["text"].astype(str)

# 3) Load Hugging Face sentiment pipeline (runs locally)
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# 4) Predict sentiment (batching helps)
texts = df["text"].tolist()
results = sentiment(texts, truncation=True)

df["hf_label"] = [r["label"] for r in results]          # POSITIVE / NEGATIVE
df["hf_score"] = [r["score"] for r in results]          # confidence
df["sentiment"] = df["hf_label"].map({"POSITIVE": "Positive", "NEGATIVE": "Negative"})

# 5) Save new file for Streamlit / Render
df.to_csv("reviews_with_sentiment.csv", index=False)
print("Saved: reviews_with_sentiment.csv")
