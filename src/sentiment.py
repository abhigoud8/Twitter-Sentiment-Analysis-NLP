import re
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

def vader_sentiment(text: str) -> str:
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"

def bert_sentiment(texts: list) -> list:
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    results = classifier(texts, truncation=True, max_length=512)
    return [r["label"] for r in results]

def analyze(df: pd.DataFrame, text_col: str = "tweet") -> pd.DataFrame:
    df["clean_text"] = df[text_col].apply(clean_text)
    df["vader_sentiment"] = df["clean_text"].apply(vader_sentiment)
    df["bert_sentiment"] = bert_sentiment(df["clean_text"].tolist())
    return df

if __name__ == "__main__":
    sample = pd.DataFrame({"tweet": [
        "I love this product! Absolutely amazing!",
        "This is the worst experience I have ever had.",
        "The weather today is okay, nothing special."
    ]})
    result = analyze(sample)
    print(result[["tweet", "vader_sentiment", "bert_sentiment"]])