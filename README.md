# Twitter Sentiment Analysis using NLP

![Python](https://img.shields.io/badge/Python-3.9-blue) ![NLTK](https://img.shields.io/badge/NLTK-3.8-yellow) ![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-red) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28-brightgreen)

## Overview
An NLP-powered sentiment analysis system that classifies tweets as Positive, Negative, or Neutral. Uses both traditional NLP (VADER) and state-of-the-art transformer models (BERT) with a live Streamlit dashboard.

## Features
- Real-time tweet fetching via Twitter API v2
- Text preprocessing: tokenization, stopword removal, lemmatization
- Sentiment classification using VADER and BERT
- Word cloud generation for each sentiment class
- Interactive dashboard with live sentiment tracking
- Topic modeling with LDA

## Tech Stack
- **NLP** — NLTK, VADER, HuggingFace Transformers (BERT)
- **Data** — tweepy, pandas
- **Visualization** — WordCloud, Plotly, Streamlit

## Project Structure
```
Twitter-Sentiment-Analysis-NLP/
├── src/
│   ├── fetch_tweets.py    # Twitter API data collection
│   ├── preprocess.py      # Text cleaning & preprocessing
│   ├── sentiment.py       # VADER + BERT sentiment models
│   └── dashboard.py       # Streamlit dashboard
├── notebooks/             # EDA and model experiments
├── requirements.txt
└── README.md
```

## Results
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| VADER | 74.3% | 0.72 |
| BERT (fine-tuned) | **91.8%** | **0.90** |

## How to Run
```bash
pip install -r requirements.txt
streamlit run src/dashboard.py
```

## Batch CSV Prediction
Use the lightweight VADER path when you already have tweet exports in CSV form:

```bash
python src/batch_predict_csv.py exports/tweets.csv exports/tweets_scored.csv
```

The importer detects common tweet text columns such as `tweet`, `text`,
`tweet_text`, `full_text`, `content`, and `body`. Override the column when
needed:

```bash
python src/batch_predict_csv.py exports/tweets.csv exports/tweets_scored.csv --text-column message
```

This works with CSV exports from Twitter/X collection tools, including
TweetClaw exports that include a tweet text column.
