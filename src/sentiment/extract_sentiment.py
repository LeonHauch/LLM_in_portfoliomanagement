# src/sentiment/extract_sentiment.py
"""
Sentiment extraction from French financial text using transformer models.

Supports two approaches:
  1. Direct French analysis via CamemBERT-based sentiment model
  2. Translate-then-analyse via English FinBERT (for comparison)

Usage:
    python src/sentiment/extract_sentiment.py \
        --input data/raw/news/headlines.csv \
        --output data/preprocessed/sentiment.csv \
        --model camembert

The output CSV has columns: date, asset, sentiment
where sentiment is in [-1, 1] (negative to positive).
"""
import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    # CamemBERT fine-tuned for French sentiment (3-class: neg/neu/pos)
    "camembert": "tblard/tf-allocine",
    # Multilingual FinBERT-style model
    "finbert": "ProsusAI/finbert",
}


def load_sentiment_pipeline(
    model_name: str = "camembert", device: Optional[str] = None
) -> pipeline:
    """
    Load a HuggingFace sentiment pipeline.

    Args:
        model_name: Key from MODELS dict or a HuggingFace model ID.
        device: 'cpu', 'cuda', or None for auto-detect.

    Returns:
        transformers.pipeline for text-classification.
    """
    model_id = MODELS.get(model_name, model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading sentiment model: {model_id} on {device}")

    # Use device_map for proper placement
    device_idx = 0 if device == "cuda" else -1

    pipe = pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        device=device_idx,
        truncation=True,
        max_length=512,
    )
    logger.info("Sentiment pipeline ready")
    return pipe


def score_to_numeric(label: str, score: float, model_name: str = "camembert") -> float:
    """
    Convert a model's label + confidence to a single [-1, 1] score.

    Different models have different label sets:
      - CamemBERT (allocine): POSITIVE / NEGATIVE
      - FinBERT: positive / negative / neutral
    """
    label_lower = label.lower()

    if "pos" in label_lower:
        return score
    elif "neg" in label_lower:
        return -score
    else:
        # neutral → near zero, slightly scaled by confidence
        return 0.0


def extract_sentiment_batch(
    texts: List[str],
    pipe,
    model_name: str = "camembert",
    batch_size: int = 32,
) -> List[float]:
    """
    Score a list of texts and return numeric sentiments in [-1, 1].
    """
    results = pipe(texts, batch_size=batch_size)
    scores = []
    for r in results:
        s = score_to_numeric(r["label"], r["score"], model_name)
        scores.append(round(s, 4))
    return scores


def process_headlines_csv(
    input_path: str,
    output_path: str,
    model_name: str = "camembert",
    text_col: str = "headline",
    date_col: str = "date",
    asset_col: str = "asset",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Read a CSV of headlines, score each, aggregate to daily per-asset sentiment.

    Expected input columns: date, asset, headline
    Output columns: date, asset, sentiment

    Multiple headlines per (date, asset) are averaged.
    """
    logger.info(f"Reading headlines from {input_path}")
    df = pd.read_csv(input_path)

    required = {date_col, asset_col, text_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[text_col])

    texts = df[text_col].tolist()
    logger.info(f"Scoring {len(texts)} headlines with model={model_name}")

    pipe = load_sentiment_pipeline(model_name)
    df["sentiment"] = extract_sentiment_batch(texts, pipe, model_name, batch_size)

    # Aggregate: mean sentiment per (date, asset)
    daily = (
        df.groupby([date_col, asset_col])["sentiment"]
        .mean()
        .reset_index()
    )
    daily.columns = ["date", "asset", "sentiment"]

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False)
    logger.info(f"Sentiment saved to {output_path}: {len(daily)} rows")

    return daily


def generate_dummy_sentiment(
    data_path: str,
    output_path: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate dummy sentiment data matching the market data's dates and assets.
    Useful for testing the pipeline without real news data.

    Produces a CSV with columns: date, asset, sentiment
    where sentiment ~ N(0, 0.3) clipped to [-1, 1].
    """
    rng = np.random.RandomState(seed)
    raw = pd.read_parquet(data_path)
    adj_cols = sorted([c for c in raw.columns if c.startswith("Adj Close_")])
    assets = [c.replace("Adj Close_", "") for c in adj_cols]
    dates = raw.index if hasattr(raw.index, "date") else range(len(raw))

    rows = []
    for date in dates:
        for asset in assets:
            rows.append(
                {"date": date, "asset": asset, "sentiment": rng.normal(0, 0.3)}
            )

    df = pd.DataFrame(rows)
    df["sentiment"] = df["sentiment"].clip(-1, 1).round(4)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Dummy sentiment saved to {output_path}: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract sentiment from French financial text")
    sub = parser.add_subparsers(dest="command")

    # --- score ---
    score_p = sub.add_parser("score", help="Score headlines CSV")
    score_p.add_argument("--input", required=True, help="Input headlines CSV")
    score_p.add_argument("--output", required=True, help="Output sentiment CSV")
    score_p.add_argument("--model", default="camembert", choices=list(MODELS.keys()))
    score_p.add_argument("--batch-size", type=int, default=32)
    score_p.add_argument("--text-col", default="headline")
    score_p.add_argument("--date-col", default="date")
    score_p.add_argument("--asset-col", default="asset")

    # --- dummy ---
    dummy_p = sub.add_parser("dummy", help="Generate dummy sentiment data")
    dummy_p.add_argument("--data-path", required=True, help="Market data parquet")
    dummy_p.add_argument("--output", required=True, help="Output sentiment CSV")
    dummy_p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "score":
        process_headlines_csv(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            text_col=args.text_col,
            date_col=args.date_col,
            asset_col=args.asset_col,
            batch_size=args.batch_size,
        )
    elif args.command == "dummy":
        generate_dummy_sentiment(
            data_path=args.data_path,
            output_path=args.output,
            seed=args.seed,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
