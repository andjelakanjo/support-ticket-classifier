from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src import config
from src.model import BertTicketClassifier, LSTMTicketClassifier
from src.preprocessing import TextPreprocessor


def set_seed(seed: int | None = None) -> None:
    """Set random seeds for reproducibility."""
    seed = seed if seed is not None else config.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Compute accuracy and macro F1 score."""
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(labels_np, predictions)),
        "f1": float(f1_score(labels_np, predictions, average="macro", zero_division=0)),
    }


def load_processed_dataloaders(
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load processed CSV splits and return train/val/test DataLoaders."""
    preprocessor = TextPreprocessor()
    train_df = pd.read_csv(config.TRAIN_FILE)
    val_df = pd.read_csv(config.VAL_FILE)
    test_df = pd.read_csv(config.TEST_FILE)

    train_dataset = preprocessor.create_dataset(train_df)
    val_dataset = preprocessor.create_dataset(val_df)
    test_dataset = preprocessor.create_dataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def plot_training_curves(
    history: list[dict[str, Any]],
    model_name: str,
    save_path: Path | None = None,
) -> None:
    """Plot training and validation loss/accuracy curves."""
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_accuracy"] for entry in history]
    val_acc = [entry["val_accuracy"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="Train")
    axes[0].plot(epochs, val_loss, marker="o", label="Validation")
    axes[0].set_title(f"{model_name.upper()} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, marker="o", label="Train")
    axes[1].plot(epochs, val_acc, marker="o", label="Validation")
    axes[1].set_title(f"{model_name.upper()} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


class Trainer:
    """Universal training loop for ticket classification models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        device: torch.device,
        checkpoint_path: Path,
        patience: int | None = None,
        model_name: str = "model",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.patience = patience if patience is not None else config.EARLY_STOPPING_PATIENCE
        self.model_name = model_name
        self.history: list[dict[str, Any]] = []
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch and return aggregated metrics."""
        self.model.train()
        total_loss = 0.0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for batch in self.train_loader:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad()
            outputs = self.model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
            loss = outputs["loss"]
            assert loss is not None
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_logits.append(outputs["logits"].detach())
            all_labels.append(batch["labels"])

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(logits, labels)
        metrics["loss"] = total_loss / len(self.train_loader)
        return metrics

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the model on a dataloader."""
        self.model.eval()
        total_loss = 0.0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch(batch)
                outputs = self.model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["labels"],
                )
                loss = outputs["loss"]
                assert loss is not None
                total_loss += loss.item()
                all_logits.append(outputs["logits"])
                all_labels.append(batch["labels"])

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(logits, labels)
        metrics["loss"] = total_loss / len(dataloader)
        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
        path: Path | None = None,
    ) -> None:
        """Save model, optimizer, and scheduler state."""
        save_path = path or self.checkpoint_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "model_name": self.model_name,
        }
        torch.save(checkpoint, save_path)
        if is_best:
            print(f"Saved best checkpoint to {save_path}")

    def load_checkpoint(self, path: Path | None = None) -> dict[str, Any]:
        """Load a checkpoint and restore training state."""
        load_path = path or self.checkpoint_path
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint

    def _save_history(self) -> None:
        config.TRAINING_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        history_path = config.TRAINING_HISTORY_DIR / f"{self.model_name}_history.json"
        with history_path.open("w", encoding="utf-8") as history_file:
            json.dump(self.history, history_file, indent=2)

    def fit(self, num_epochs: int) -> list[dict[str, Any]]:
        """Run the full training loop with early stopping."""
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step(val_metrics["loss"])

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "lr": current_lr,
            }
            self.history.append(epoch_record)
            self._save_history()

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | lr={current_lr:.2e}"
            )

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        return self.history


def train_lstm(
    device: torch.device | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> tuple[Trainer, list[dict[str, Any]]]:
    """Train the LSTM baseline model."""
    device = device or get_device()
    batch_size = batch_size or config.LSTM_TRAIN_BATCH_SIZE
    learning_rate = learning_rate or config.LSTM_TRAIN_LR
    num_epochs = num_epochs or config.LSTM_TRAIN_EPOCHS

    train_loader, val_loader, _ = load_processed_dataloaders(batch_size)
    model = LSTMTicketClassifier(num_classes=config.NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_path=config.LSTM_BEST_MODEL_PATH,
        patience=config.EARLY_STOPPING_PATIENCE,
        model_name="lstm",
    )
    history = trainer.fit(num_epochs=num_epochs)
    return trainer, history


def train_bert(
    device: torch.device | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> tuple[Trainer, list[dict[str, Any]]]:
    """Train the BERT fine-tuned model."""
    device = device or get_device()
    batch_size = batch_size or config.BERT_TRAIN_BATCH_SIZE
    learning_rate = learning_rate or config.BERT_TRAIN_LR
    num_epochs = num_epochs or config.BERT_TRAIN_EPOCHS

    train_loader, val_loader, _ = load_processed_dataloaders(batch_size)
    model = BertTicketClassifier(num_classes=config.NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_path=config.BERT_BEST_MODEL_PATH,
        patience=config.EARLY_STOPPING_PATIENCE,
        model_name="bert",
    )
    history = trainer.fit(num_epochs=num_epochs)
    return trainer, history
