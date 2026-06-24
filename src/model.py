from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, BertModel, PreTrainedTokenizer

from src import config


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts for a model."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}


def load_glove_embeddings(
    path: Path,
    tokenizer: PreTrainedTokenizer,
    embedding_dim: int,
) -> torch.Tensor | None:
    """Load GloVe vectors and map them onto tokenizer vocabulary indices."""
    if not path.exists():
        return None

    glove_vectors: dict[str, torch.Tensor] = {}
    with path.open("r", encoding="utf-8") as glove_file:
        for line in glove_file:
            parts = line.rstrip().split(" ")
            if len(parts) <= embedding_dim:
                continue
            word = parts[0]
            vector = torch.tensor([float(value) for value in parts[1 : embedding_dim + 1]])
            glove_vectors[word] = vector

    vocab_size = len(tokenizer)
    embedding_matrix = torch.empty(vocab_size, embedding_dim)
    nn.init.uniform_(embedding_matrix, -0.05, 0.05)

    special_tokens = set(tokenizer.all_special_tokens)
    matched = 0
    for token_id in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token in special_tokens:
            continue
        word = token.replace("##", "")
        if word in glove_vectors:
            embedding_matrix[token_id] = glove_vectors[word]
            matched += 1

    print(f"GloVe: mapped {matched}/{vocab_size} tokenizer entries from {path.name}")
    return embedding_matrix


class LSTMTicketClassifier(nn.Module):
    """Bidirectional LSTM baseline for support ticket classification."""

    def __init__(
        self,
        num_classes: int,
        vocab_size: int | None = None,
        embedding_dim: int | None = None,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
        glove_path: Path | None = None,
        tokenizer_name: str | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim or config.LSTM_EMBEDDING_DIM
        self.hidden_dim = hidden_dim or config.LSTM_HIDDEN_DIM
        self.num_layers = num_layers or config.LSTM_NUM_LAYERS
        self.dropout_rate = dropout if dropout is not None else config.LSTM_DROPOUT
        self.tokenizer_name = tokenizer_name or config.TOKENIZER_NAME
        self.glove_path = glove_path or config.GLOVE_PATH

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.vocab_size = vocab_size or len(tokenizer)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=tokenizer.pad_token_id)
        glove_matrix = load_glove_embeddings(self.glove_path, tokenizer, self.embedding_dim)
        if glove_matrix is not None:
            self.embedding.weight.data.copy_(glove_matrix)
        self.embedding_init = "glove" if glove_matrix is not None else "random"

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_dim * 2, num_classes)

    def _masked_mean_pool(
        self,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return sequence.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).float()
        summed = (sequence * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return summed / lengths

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        embedded = self.embedding(input_ids)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = pack_padded_sequence(
                embedded,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.lstm(packed)
            lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_output, _ = self.lstm(embedded)

        pooled = self._masked_mean_pool(
            lstm_output,
            attention_mask[:, : lstm_output.size(1)] if attention_mask is not None else None,
        )
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        probs = F.softmax(logits, dim=-1)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"logits": logits, "probs": probs, "loss": loss}

    def count_parameters(self) -> dict[str, int]:
        return count_parameters(self)

    def describe_architecture(self) -> str:
        return (
            "LSTMTicketClassifier\n"
            f"- Embedding: {self.vocab_size} x {self.embedding_dim} ({self.embedding_init} init)\n"
            f"- BiLSTM: {self.num_layers} layers, hidden_dim={self.hidden_dim}, bidirectional=True\n"
            f"- Dropout: p={self.dropout_rate}\n"
            f"- Classifier: Linear({self.hidden_dim * 2} -> {self.num_classes})\n"
            f"- Output: logits + softmax over {self.num_classes} classes"
        )

    def parameter_groups(self) -> dict[str, int]:
        return {
            "embedding": count_parameters(self.embedding)["total"],
            "lstm": count_parameters(self.lstm)["total"],
            "classifier": count_parameters(self.classifier)["total"],
        }


class BertTicketClassifier(nn.Module):
    """Fine-tuned BERT classifier for support ticket classification."""

    def __init__(
        self,
        num_classes: int,
        model_name: str | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name or config.MODEL_NAME
        self.dropout_rate = dropout if dropout is not None else config.BERT_DROPOUT

        self.bert = BertModel.from_pretrained(self.model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        probs = F.softmax(logits, dim=-1)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"logits": logits, "probs": probs, "loss": loss}

    def count_parameters(self) -> dict[str, int]:
        return count_parameters(self)

    def describe_architecture(self) -> str:
        hidden_size = self.bert.config.hidden_size
        return (
            "BertTicketClassifier\n"
            f"- Encoder: {self.model_name} (hidden_size={hidden_size})\n"
            f"- Dropout: p={self.dropout_rate}\n"
            f"- Classifier: Linear({hidden_size} -> {self.num_classes})\n"
            f"- Output: logits + softmax over {self.num_classes} classes"
        )

    def parameter_groups(self) -> dict[str, int]:
        return {
            "bert": count_parameters(self.bert)["total"],
            "classifier": count_parameters(self.classifier)["total"],
        }
