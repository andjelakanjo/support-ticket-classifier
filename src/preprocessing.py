from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src import config
from src.data_loader import load_raw_dataset, to_dataframe

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-z0-9\s.,!?;:'\"()-]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def load_dataframe(path: Path | None = None) -> pd.DataFrame:
    """Load normalized ticket data from cache or HuggingFace."""
    data_path = path or config.RAW_DATA_FILE
    if data_path.exists():
        return pd.read_csv(data_path)

    dataset = load_raw_dataset()
    return to_dataframe(dataset, language=config.LANGUAGE_FILTER, save_cache=True)


class TicketDataset(Dataset):
    """PyTorch Dataset for tokenized support tickets."""

    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TextPreprocessor:
    """End-to-end text preprocessing pipeline for support ticket classification."""

    def __init__(
        self,
        tokenizer_name: str | None = None,
        max_length: int | None = None,
        label2id: dict[str, int] | None = None,
    ) -> None:
        self.tokenizer_name = tokenizer_name or config.TOKENIZER_NAME
        self.max_length = max_length or config.MAX_SEQUENCE_LENGTH
        self.label2id = label2id or config.LABEL2ID
        self._tokenizer: PreTrainedTokenizer | None = None

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    def clean_text(self, text: str) -> str:
        """Remove HTML, URLs, special characters, and normalize whitespace."""
        if not isinstance(text, str):
            text = str(text or "")

        cleaned = html.unescape(text)
        cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
        cleaned = URL_PATTERN.sub(" ", cleaned)
        cleaned = cleaned.lower()
        cleaned = SPECIAL_CHAR_PATTERN.sub(" ", cleaned)
        cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
        return cleaned

    def tokenize(
        self,
        texts: list[str] | str,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Tokenize text(s) using the HuggingFace tokenizer."""
        if isinstance(texts, str):
            texts = [texts]

        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

    def encode_labels(self, labels: list[str]) -> list[int]:
        """Encode string category labels into numeric IDs."""
        encoded: list[int] = []
        for label in labels:
            if label not in self.label2id:
                raise ValueError(f"Unknown label: {label!r}")
            encoded.append(self.label2id[label])
        return encoded

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        test_ratio: float | None = None,
        seed: int | None = None,
        save: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into stratified train/validation/test sets."""
        train_ratio = train_ratio if train_ratio is not None else config.TRAIN_RATIO
        val_ratio = val_ratio if val_ratio is not None else config.VAL_RATIO
        test_ratio = test_ratio if test_ratio is not None else config.TEST_RATIO
        seed = seed if seed is not None else config.RANDOM_SEED

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        labels = df[config.LABEL_COLUMN]
        train_df, temp_df = train_test_split(
            df,
            test_size=(1.0 - train_ratio),
            random_state=seed,
            stratify=labels,
        )

        relative_test_size = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            random_state=seed,
            stratify=temp_df[config.LABEL_COLUMN],
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        if save:
            config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(config.TRAIN_FILE, index=False)
            val_df.to_csv(config.VAL_FILE, index=False)
            test_df.to_csv(config.TEST_FILE, index=False)

        return train_df, val_df, test_df

    def create_dataset(self, df: pd.DataFrame) -> TicketDataset:
        """Create a PyTorch Dataset from a normalized DataFrame."""
        texts = [self.clean_text(text) for text in df[config.TEXT_COLUMN].tolist()]
        labels = self.encode_labels(df[config.LABEL_COLUMN].tolist())

        encodings = self.tokenize(texts, return_tensors=None)
        encoding_dict = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
        return TicketDataset(encoding_dict, labels)

    def preprocess_pipeline(
        self,
        path: Path | None = None,
        save_splits: bool = True,
    ) -> tuple[TicketDataset, TicketDataset, TicketDataset]:
        """Run the full preprocessing pipeline and return train/val/test datasets."""
        df = load_dataframe(path)
        train_df, val_df, test_df = self.split_data(df, save=save_splits)
        return (
            self.create_dataset(train_df),
            self.create_dataset(val_df),
            self.create_dataset(test_df),
        )
