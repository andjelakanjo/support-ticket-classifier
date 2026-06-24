from pathlib import Path

# Reproducibility
RANDOM_SEED = 42

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
RAW_DATA_FILE = RAW_DATA_DIR / "tickets.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pt"

# HuggingFace dataset
HF_DATASET_NAME = "Tobi-Bueck/customer-support-tickets"
HF_DATASET_SPLIT = "train"
HF_TEXT_COLUMNS = ("subject", "body")
HF_LABEL_COLUMN = "queue"
LANGUAGE_FILTER = "en"

# Dataset columns
TEXT_COLUMN = "text"
LABEL_COLUMN = "category"

# Ticket categories (queue departments from the dataset)
TICKET_CATEGORIES = [
    "Technical Support",
    "Customer Service",
    "Billing and Payments",
    "Product Support",
    "IT Support",
    "Returns and Exchanges",
    "Sales and Pre-Sales",
    "Human Resources",
    "Service Outages and Maintenance",
    "General Inquiry",
]
NUM_CLASSES = len(TICKET_CATEGORIES)
LABEL2ID = {label: i for i, label in enumerate(TICKET_CATEGORIES)}
ID2LABEL = {i: label for i, label in enumerate(TICKET_CATEGORIES)}

# Pretrained model
MODEL_NAME = "bert-base-uncased"
TOKENIZER_NAME = "bert-base-uncased"
MAX_SEQUENCE_LENGTH = 128

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training hyperparameters
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 3

# LSTM baseline
LSTM_EMBEDDING_DIM = 100
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
GLOVE_PATH = PROJECT_ROOT / "data" / "embeddings" / "glove.6B.100d.txt"
LSTM_TRAIN_LR = 1e-3
LSTM_TRAIN_BATCH_SIZE = 32
LSTM_TRAIN_EPOCHS = 20
LSTM_BEST_MODEL_PATH = MODELS_DIR / "best_lstm.pt"

# BERT classifier head
BERT_DROPOUT = 0.3
BERT_TRAIN_LR = 2e-5
BERT_TRAIN_BATCH_SIZE = 16
BERT_TRAIN_EPOCHS = 5
BERT_BEST_MODEL_PATH = MODELS_DIR / "best_bert.pt"

TRAINING_HISTORY_DIR = LOGS_DIR / "training"
RESULTS_DIR = PROJECT_ROOT / "results"

# Hyperparameter optimization (Optuna)
HPO_N_TRIALS = 20
HPO_MAX_EPOCHS = 3
HPO_RESULTS_CSV = RESULTS_DIR / "hyperparameter_search.csv"
HPO_VISUALIZATIONS_DIR = RESULTS_DIR / "hpo"
HPO_STUDY_NAME = "lstm_hyperparameter_search"
