from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from src import config
from src.model import BertTicketClassifier, LSTMTicketClassifier
from src.preprocessing import TextPreprocessor
from src.train import get_device

ModelType = Literal["lstm", "bert"]


@dataclass
class EvaluationResult:
    """Container for model evaluation outputs."""

    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_probs: np.ndarray
    texts: list[str]
    summary: dict[str, float]


def load_trained_model(
    model_type: ModelType,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module | None:
    """Load a trained model from checkpoint. Returns None if checkpoint is missing."""
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    if model_type == "lstm":
        model = LSTMTicketClassifier(num_classes=config.NUM_CLASSES)
    else:
        model = BertTicketClassifier(num_classes=config.NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_test_loader_and_texts(
    batch_size: int | None = None,
) -> tuple[DataLoader, list[str]]:
    """Create test DataLoader and aligned text list from processed test CSV."""
    batch_size = batch_size or config.EVAL_BATCH_SIZE
    preprocessor = TextPreprocessor()
    test_df = pd.read_csv(config.TEST_FILE)
    texts = test_df[config.TEXT_COLUMN].astype(str).tolist()
    test_dataset = preprocessor.create_dataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, texts


def run_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return true labels, predictions, and probabilities."""
    model.eval()
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask)
            probs = outputs["probs"].cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(probs)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    return y_true, y_pred, y_probs


def compute_summary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> dict[str, float]:
    """Compute overall evaluation metrics."""
    y_true_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))
    try:
        macro_auroc = float(
            roc_auc_score(y_true_bin, y_probs, average="macro", multi_class="ovr")
        )
    except ValueError:
        macro_auroc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_auroc": macro_auroc,
    }


class ModelEvaluator:
    """Run full evaluation pipeline for a single model."""

    def __init__(
        self,
        model_name: ModelType,
        results_dir: Path | None = None,
        max_error_examples: int = 50,
    ) -> None:
        self.model_name = model_name
        self.results_dir = (results_dir or config.RESULTS_DIR) / model_name
        self.max_error_examples = max_error_examples
        self.class_names = config.TICKET_CATEGORIES

    def _save_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        rows = []
        for label_name, metrics in report_dict.items():
            if isinstance(metrics, dict):
                rows.append(
                    {
                        "class": label_name,
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1-score": metrics.get("f1-score"),
                        "support": metrics.get("support"),
                    }
                )
        report_df = pd.DataFrame(rows)
        report_df.to_csv(self.results_dir / "classification_report.csv", index=False)
        return report_df

    def _save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        matrix = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
        matrix_df = pd.DataFrame(matrix, index=self.class_names, columns=self.class_names)
        matrix_df.to_csv(self.results_dir / "confusion_matrix.csv")

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix_df,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
        )
        plt.title(f"Confusion Matrix — {self.model_name.upper()}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.results_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _save_roc_curves(self, y_true: np.ndarray, y_probs: np.ndarray) -> pd.DataFrame:
        y_true_bin = label_binarize(y_true, classes=list(range(config.NUM_CLASSES)))
        auc_rows = []

        plt.figure(figsize=(10, 8))
        valid_aucs: list[float] = []

        for class_idx, class_name in enumerate(self.class_names):
            if y_true_bin[:, class_idx].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_probs[:, class_idx])
            class_auc = auc(fpr, tpr)
            valid_aucs.append(class_auc)
            auc_rows.append({"class": class_name, "auc": class_auc})
            plt.plot(fpr, tpr, label=f"{class_name} (AUC={class_auc:.3f})")

        macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.title(f"ROC Curves (OvR) — {self.model_name.upper()} | macro AUC={macro_auc:.3f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig(self.results_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
        plt.close()

        auc_df = pd.DataFrame(auc_rows)
        auc_df.to_csv(self.results_dir / "roc_auc.csv", index=False)
        return auc_df

    def _save_misclassified_examples(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        texts: list[str],
    ) -> pd.DataFrame:
        misclassified_rows = []
        for idx, (true_id, pred_id, probs) in enumerate(zip(y_true, y_pred, y_probs)):
            if true_id == pred_id:
                continue
            true_label = config.ID2LABEL[int(true_id)]
            pred_label = config.ID2LABEL[int(pred_id)]
            confidence = float(np.max(probs))
            text = texts[idx]
            text_preview = text[:300] + ("..." if len(text) > 300 else "")
            explanation = (
                f"Model je predvideo '{pred_label}' (pouzdanost {confidence:.2%}), "
                f"ali prava kategorija je '{true_label}'."
            )
            misclassified_rows.append(
                {
                    "text": text_preview,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": confidence,
                    "explanation": explanation,
                }
            )

        errors_df = pd.DataFrame(misclassified_rows)
        if not errors_df.empty:
            errors_df = errors_df.sort_values("confidence", ascending=False).head(
                self.max_error_examples
            )
        errors_df.to_csv(self.results_dir / "misclassified_examples.csv", index=False)
        return errors_df

    def _save_summary_metrics(self, summary: dict[str, float]) -> None:
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(self.results_dir / "summary_metrics.csv", index=False)

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        texts: list[str],
        device: torch.device,
    ) -> EvaluationResult:
        """Run full evaluation and save all artifacts."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        y_true, y_pred, y_probs = run_predictions(model, test_loader, device)
        summary = compute_summary_metrics(y_true, y_pred, y_probs)

        self._save_classification_report(y_true, y_pred)
        self._save_confusion_matrix(y_true, y_pred)
        self._save_roc_curves(y_true, y_probs)
        self._save_misclassified_examples(y_true, y_pred, y_probs, texts)
        self._save_summary_metrics(summary)

        print(
            f"[{self.model_name.upper()}] accuracy={summary['accuracy']:.4f} "
            f"macro_f1={summary['macro_f1']:.4f} macro_auroc={summary['macro_auroc']:.4f}"
        )

        return EvaluationResult(
            model_name=self.model_name,
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            texts=texts,
            summary=summary,
        )


def compare_models(results: dict[str, EvaluationResult]) -> pd.DataFrame:
    """Build comparison table for evaluated models."""
    rows = []
    for model_name, result in results.items():
        rows.append(
            {
                "model": model_name.upper(),
                "accuracy": result.summary["accuracy"],
                "macro_f1": result.summary["macro_f1"],
                "weighted_f1": result.summary["weighted_f1"],
                "macro_auroc": result.summary["macro_auroc"],
            }
        )
    comparison_df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)
    return comparison_df


def evaluate_model(
    model_type: ModelType,
    checkpoint_path: Path,
    test_loader: DataLoader,
    texts: list[str],
    device: torch.device,
    results_dir: Path | None = None,
) -> EvaluationResult | None:
    """Evaluate a single model if checkpoint exists."""
    model = load_trained_model(model_type, checkpoint_path, device)
    if model is None:
        return None

    evaluator = ModelEvaluator(model_name=model_type, results_dir=results_dir)
    return evaluator.evaluate(model, test_loader, texts, device)


def evaluate_all(
    lstm_checkpoint: Path | None = None,
    bert_checkpoint: Path | None = None,
    results_dir: Path | None = None,
    batch_size: int | None = None,
) -> dict[str, EvaluationResult | None]:
    """Evaluate all available trained models on the test set."""
    lstm_checkpoint = lstm_checkpoint or config.LSTM_BEST_MODEL_PATH
    bert_checkpoint = bert_checkpoint or config.BERT_BEST_MODEL_PATH
    results_dir = results_dir or config.RESULTS_DIR
    device = get_device()

    test_loader, texts = get_test_loader_and_texts(batch_size=batch_size)
    evaluated: dict[str, EvaluationResult | None] = {}

    evaluated["lstm"] = evaluate_model(
        "lstm", lstm_checkpoint, test_loader, texts, device, results_dir
    )
    if evaluated["lstm"] is None and lstm_checkpoint.exists() is False:
        print("LSTM checkpoint is missing. Train LSTM before evaluation.")

    if bert_checkpoint.exists():
        evaluated["bert"] = evaluate_model(
            "bert", bert_checkpoint, test_loader, texts, device, results_dir
        )
    else:
        print(f"Warning: BERT checkpoint not found at {bert_checkpoint}. Skipping BERT evaluation.")
        evaluated["bert"] = None

    successful = {name: result for name, result in evaluated.items() if result is not None}
    if len(successful) >= 2:
        compare_models(successful)
    elif len(successful) == 1:
        only_name = next(iter(successful))
        print(f"Only {only_name.upper()} evaluated. Model comparison requires at least two models.")

    return evaluated
