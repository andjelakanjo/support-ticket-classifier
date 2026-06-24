from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import optuna
import pandas as pd
import torch
from optuna.study import Study
from torch.utils.data import DataLoader

from src import config
from src.model import LSTMTicketClassifier
from src.preprocessing import TextPreprocessor, TicketDataset
from src.train import Trainer, get_device, set_seed

_CACHED_DATASETS: tuple[TicketDataset, TicketDataset] | None = None


def _get_cached_datasets() -> tuple[TicketDataset, TicketDataset]:
    """Load and tokenize train/val datasets once per process."""
    global _CACHED_DATASETS
    if _CACHED_DATASETS is None:
        preprocessor = TextPreprocessor()
        train_df = pd.read_csv(config.TRAIN_FILE)
        val_df = pd.read_csv(config.VAL_FILE)
        _CACHED_DATASETS = (
            preprocessor.create_dataset(train_df),
            preprocessor.create_dataset(val_df),
        )
    return _CACHED_DATASETS


def _create_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders from cached datasets."""
    train_dataset, val_dataset = _get_cached_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def create_objective(device: torch.device) -> Callable[[optuna.Trial], float]:
    """Create Optuna objective function for LSTM hyperparameter search."""

    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 3)

        set_seed(config.RANDOM_SEED + trial.number)
        train_loader, val_loader = _create_dataloaders(batch_size)

        model = LSTMTicketClassifier(
            num_classes=config.NUM_CLASSES,
            dropout=dropout,
            num_layers=num_lstm_layers,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1
        )

        checkpoint_path = config.MODELS_DIR / f"hpo_trial_{trial.number}.pt"
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_path=checkpoint_path,
            patience=999,
            model_name=f"hpo_trial_{trial.number}",
        )

        history = trainer.fit(num_epochs=config.HPO_MAX_EPOCHS)
        val_f1 = history[-1]["val_f1"]

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        return float(val_f1)

    return objective


def save_study_results(study: Study, path: Path | None = None) -> pd.DataFrame:
    """Save Optuna study results to CSV."""
    output_path = path or config.HPO_RESULTS_CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = study.trials_dataframe()
    results_df.to_csv(output_path, index=False)
    return results_df


def _save_plotly_figure(fig, output_path: Path) -> None:
    """Save Plotly figure as PNG with HTML fallback."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(output_path))
    except Exception:
        html_path = output_path.with_suffix(".html")
        fig.write_html(str(html_path))
        print(f"PNG export failed, saved HTML instead: {html_path}")


def plot_hpo_visualizations(
    study: Study,
    output_dir: Path | None = None,
) -> None:
    """Generate and save Optuna visualization plots."""
    output_dir = output_dir or config.HPO_VISUALIZATIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        print("No completed trials available for visualization.")
        return

    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
        )
    except ImportError as exc:
        print(f"Skipping visualizations (plotly not available): {exc}")
        return

    plot_specs = [
        ("param_importance.png", plot_param_importances),
        ("parallel_coordinate.png", plot_parallel_coordinate),
        ("optimization_history.png", plot_optimization_history),
    ]
    for filename, plot_fn in plot_specs:
        try:
            _save_plotly_figure(plot_fn(study), output_dir / filename)
        except Exception as exc:
            print(f"Could not generate {filename}: {exc}")


def run_hyperparameter_search(
    n_trials: int | None = None,
    device: torch.device | None = None,
    study_name: str | None = None,
) -> Study:
    """Run Optuna hyperparameter search for the LSTM model."""
    n_trials = n_trials if n_trials is not None else config.HPO_N_TRIALS
    device = device or get_device()
    study_name = study_name or config.HPO_STUDY_NAME

    set_seed(config.RANDOM_SEED)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
    )

    print(f"Starting hyperparameter search: {n_trials} trials on {device}")
    study.optimize(create_objective(device), n_trials=n_trials, show_progress_bar=True)

    save_study_results(study)
    plot_hpo_visualizations(study)

    print("\nBest trial:")
    print(f"  Value (val macro F1): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    return study
