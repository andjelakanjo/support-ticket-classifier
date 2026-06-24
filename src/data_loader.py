from __future__ import annotations

from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from src import config

IMBALANCE_RATIO_THRESHOLD = 1.5


def load_raw_dataset() -> Dataset | DatasetDict:
    """Load the customer support tickets dataset from HuggingFace."""
    return load_dataset(config.HF_DATASET_NAME, split=config.HF_DATASET_SPLIT)


def build_text(row: dict[str, Any]) -> str:
    """Combine subject and body into a single text field."""
    parts = [str(row.get(col, "") or "").strip() for col in config.HF_TEXT_COLUMNS]
    return " ".join(part for part in parts if part)


def to_dataframe(
    dataset: Dataset | DatasetDict,
    language: str | None = None,
    save_cache: bool = True,
) -> pd.DataFrame:
    """Convert the HuggingFace dataset to a normalized pandas DataFrame."""
    if isinstance(dataset, DatasetDict):
        if config.HF_DATASET_SPLIT in dataset:
            dataset = dataset[config.HF_DATASET_SPLIT]
        else:
            dataset = next(iter(dataset.values()))

    df = dataset.to_pandas()

    if language is not None:
        df = df[df["language"].str.lower() == language.lower()].copy()

    df[config.TEXT_COLUMN] = df.apply(build_text, axis=1)
    df[config.LABEL_COLUMN] = df[config.HF_LABEL_COLUMN].astype(str).str.strip()
    df = df[df[config.TEXT_COLUMN].str.len() > 0].copy()
    df = df.reset_index(drop=True)

    if save_cache:
        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        export_cols = [config.TEXT_COLUMN, config.LABEL_COLUMN, *config.HF_TEXT_COLUMNS]
        export_cols = [col for col in export_cols if col in df.columns]
        df[export_cols].to_csv(config.RAW_DATA_FILE, index=False)

    return df


def get_basic_info(df: pd.DataFrame) -> dict[str, Any]:
    """Return basic dataset statistics."""
    return {
        "num_examples": len(df),
        "num_classes": df[config.LABEL_COLUMN].nunique(),
        "columns": list(df.columns),
        "categories": sorted(df[config.LABEL_COLUMN].unique().tolist()),
    }


def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    """Return the number of examples per class."""
    return df[config.LABEL_COLUMN].value_counts().sort_values(ascending=False)


def get_text_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for text length in characters and words."""
    char_lengths = df[config.TEXT_COLUMN].str.len()
    word_lengths = df[config.TEXT_COLUMN].str.split().str.len()

    stats = pd.DataFrame(
        {
            "characters": char_lengths,
            "words": word_lengths,
        }
    )
    return stats.describe().round(2)


def compute_imbalance_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute class imbalance metrics for the dataset."""
    distribution = get_class_distribution(df)
    max_class = distribution.idxmax()
    min_class = distribution.idxmin()
    max_count = int(distribution.max())
    min_count = int(distribution.min())
    ratio = max_count / min_count if min_count > 0 else float("inf")
    total = len(df)

    return {
        "max_class": max_class,
        "min_class": min_class,
        "max_count": max_count,
        "min_count": min_count,
        "imbalance_ratio": round(ratio, 2),
        "max_class_pct": round(100 * max_count / total, 2),
        "min_class_pct": round(100 * min_count / total, 2),
        "is_imbalanced": ratio > IMBALANCE_RATIO_THRESHOLD,
        "distribution": distribution,
    }


def get_examples_per_class(df: pd.DataFrame, n: int = 1) -> dict[str, list[str]]:
    """Return up to n example texts for each category."""
    examples: dict[str, list[str]] = {}
    for category in sorted(df[config.LABEL_COLUMN].unique()):
        subset = df[df[config.LABEL_COLUMN] == category].head(n)
        examples[category] = subset[config.TEXT_COLUMN].tolist()
    return examples
