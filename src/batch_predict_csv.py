import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from sentiment import clean_text, vader_sentiment


TEXT_COLUMN_CANDIDATES = (
    "tweet",
    "text",
    "tweet_text",
    "full_text",
    "content",
    "body",
)


def find_text_column(columns: pd.Index) -> str:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    available = ", ".join(str(column) for column in columns)
    expected = ", ".join(TEXT_COLUMN_CANDIDATES)
    raise ValueError(f"CSV needs one of these text columns: {expected}. Found: {available}")


def analyze_csv(input_path: Path, output_path: Path, text_column: Optional[str] = None) -> None:
    frame = pd.read_csv(input_path)
    column = text_column or find_text_column(frame.columns)

    output = frame.copy()
    output["clean_text"] = output[column].fillna("").astype(str).map(clean_text)
    output["vader_sentiment"] = output["clean_text"].map(vader_sentiment)
    output.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VADER sentiment analysis over a tweet CSV export."
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV with tweet text.")
    parser.add_argument("output_csv", type=Path, help="Output CSV with predictions.")
    parser.add_argument(
        "--text-column",
        help="Optional text column override. Defaults to common tweet export names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_csv(args.input_csv, args.output_csv, args.text_column)


if __name__ == "__main__":
    main()
