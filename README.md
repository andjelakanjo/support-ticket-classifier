# Klasifikacija support ticketa pomoću neuronskih mreža

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-red.svg)](https://pytorch.org/)

Automatska klasifikacija korisničkih support tiketa u odgovarajući queue departman pomoću dubokih neuronskih mreža. Projekat implementira kompletan ML pipeline — od istraživanja podataka do evaluacije — sa dva modela: LSTM baseline (BiLSTM) i BERT fine-tuning (`bert-base-uncased`).

---

## 1. Opis problema

U velikim organizacijama, korisnički support tiketi svakodnevno stižu u velikom broju i moraju se ručno prosleđivati odgovarajućim departmanima (npr. tehnička podrška, naplata, prodaja). Ručno rutiranje je sporo, skupo i podložno greškama.

**Cilj projekta** je automatska klasifikacija teksta tiketa u jednu od **10 kategorija** (queue departmana), čime se smanjuje vreme obrade i poboljšava korisničko iskustvo.

Problem se formalizuje kao **multi-class klasifikacija teksta**: na ulazu je tekst tiketa (naslov + telo poruke), na izlazu predviđena kategorija departmana.

```mermaid
flowchart LR
    ticket[SupportTicket] --> preprocess[Preprocesiranje]
    preprocess --> model[NeuronskaMreza]
    model --> queue[QueueDepartman]
```

---

## 2. Podaci

### Izvor

Podaci potiču sa HuggingFace platforme, dataset **[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)**. Za ovaj projekat korišćeni su isključivo tiketi na engleskom jeziku (`language=en`), ukupno **28.261 primera**.

Učitavanje i keširanje obavlja modul `src/data_loader.py`. Pri prvom pokretanju dataset se preuzima sa HuggingFace-a i čuva lokalno u `data/raw/tickets.csv`.

### Struktura

Originalne kolone `subject` i `body` spajaju se u jednu kolonu `text`. Kolona `queue` mapira se na `category` (ciljna promenljiva). Konačan skup sadrži 10 klasa:

| Kategorija | Broj primera |
|------------|-------------|
| Technical Support | 8.149 |
| Product Support | 5.305 |
| Customer Service | 4.269 |
| IT Support | 3.333 |
| Billing and Payments | 2.897 |
| Returns and Exchanges | 1.402 |
| Service Outages and Maintenance | 1.106 |
| Sales and Pre-Sales | 843 |
| Human Resources | 553 |
| General Inquiry | 404 |

### Analiza

Detaljna eksplorativna analiza podataka nalazi se u notebooku `notebooks/01_data_exploration.ipynb`. Ključni nalazi:

- **Broj primera:** 28.261 tiketa, **10 klasa**
- **Nebalans klasa:** odnos najveće i najmanje klase je ~**20,2:1** (Technical Support vs General Inquiry)
- **Dužina teksta:** prosečno ~410 karaktera (~60 reči), medijana ~404 karaktera
- **Distribucija:** Technical Support čini ~29% celokupnog skupa, dok General Inquiry ima samo ~1,4%

Ovakav nebalans značajno utiče na metrike klasifikacije, posebno macro F1.

### Preprocesiranje

Preprocesiranje (`src/preprocessing.py`, notebook `02_preprocessing.ipynb`) obuhvata:

1. **Čišćenje teksta** — uklanjanje HTML tagova, URL-ova i specijalnih karaktera
2. **Normalizacija** — lowercase, uklanjanje višestrukih razmaka
3. **Tokenizacija** — HuggingFace tokenizer `bert-base-uncased`, maksimalna dužina sekvence 128 tokena
4. **Stratifikovana podela** — 70% trening / 15% validacija / 15% test (`random_state=42`)

| Skup | Broj primera |
|------|-------------|
| Trening | 19.782 |
| Validacija | 4.239 |
| Test | 4.240 |

Obrađeni skupovi čuvaju se u `data/processed/train.csv`, `val.csv` i `test.csv`.

---

## 3. Arhitektura modela

Implementirana su dva modela u `src/model.py` (detalji u `notebooks/03_model_architecture.ipynb`).

### LSTMTicketClassifier (baseline)

```
input_ids (batch, seq_len)
    │
    ▼
Embedding (30522 × 100)     ← random init ili GloVe (ako je dostupan)
    │
    ▼
BiLSTM (2 sloja, hidden=256, dropout=0.3)
    │
    ▼
Masked Mean Pooling
    │
    ▼
Dropout (p=0.3)
    │
    ▼
Linear (512 → 10) → softmax
```

- **Embedding:** 30.522 × 100 (vocab BERT tokenizera); GloVe inicijalizacija ako postoji fajl na `data/embeddings/glove.6B.100d.txt`, inače slučajna inicijalizacija
- **BiLSTM:** 2 sloja, hidden dimenzija 256, bidirekcionalan
- **Pooling:** masked mean preko validnih tokena
- **Klasifikator:** Linear(512 → 10)

### BertTicketClassifier

```
input_ids + attention_mask
    │
    ▼
bert-base-uncased (768-dim pooler output)
    │
    ▼
Dropout (p=0.3)
    │
    ▼
Linear (768 → 10) → softmax
```

- **Encoder:** pretrenirani `bert-base-uncased` (110M parametara)
- **Klasifikator:** dropout + linearni sloj na pooler output
- Fine-tuning celog BERT modela za klasifikaciju tiketa

Oba modela vraćaju `{logits, probs, loss}` i koriste cross-entropy gubitak.

---

## 4. Trening

Trening je implementiran u `src/train.py` (notebook `04_training.ipynb`). Koristi se univerzalna `Trainer` klasa sa early stopping-om i `ReduceLROnPlateau` scheduler-om.

| Parametar | LSTM | BERT |
|-----------|------|------|
| Optimizer | Adam | AdamW |
| Learning rate | 1×10⁻³ | 2×10⁻⁵ |
| Batch size | 32 | 16 |
| Broj epoha | 20 | 5 |
| Weight decay | 0.01 | 0.01 |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Early stopping | patience = 3 | patience = 3 |
| Kriterijum za checkpoint | najmanji val loss | najmanji val loss |

Checkpointi se čuvaju u `models/best_lstm.pt` i `models/best_bert.pt` (isključeni iz git repozitorijuma).

### Pokretanje treninga

```bash
# LSTM baseline
python3 -c "from src.train import train_lstm; train_lstm()"

# BERT fine-tuning
python3 -c "from src.train import train_bert; train_bert()"
```

Istorija treninga (loss, accuracy, F1 po epohi) čuva se u `logs/training/` kao JSON fajlovi.

---

## 5. Analiza osetljivosti i hiperparametarska optimizacija

Hiperparametarska optimizacija (HPO) implementirana je pomoću biblioteke **Optuna** (`src/hyperparameter_search.py`, notebook `06_hyperparameter_optimization.ipynb`).

### Konfiguracija pretrage

| Parametar | Vrednost |
|-----------|----------|
| Algoritam | TPE sampler |
| Broj trial-a | 20 |
| Epohe po trial-u | 3 |
| Ciljna metrika | validation macro F1 (maximize) |
| Model | samo LSTM (BERT je prespor za 20 trial-a) |

### Search space

| Hiperparametar | Opseg |
|----------------|-------|
| `learning_rate` | 1×10⁻⁵ – 1×10⁻² (log scale) |
| `batch_size` | 16, 32, 64 |
| `dropout` | 0.1 – 0.5 |
| `num_lstm_layers` | 1 – 3 |

Radi ubrzanja, train/val skupovi se tokenizuju jednom po procesu i keširaju (`_get_cached_datasets`). Checkpointi trial-a čuvaju se privremeno kao `models/hpo_trial_{n}.pt` i brišu se nakon svakog trial-a.

### Pokretanje HPO

```bash
python3 -c "from src.hyperparameter_search import run_hyperparameter_search; run_hyperparameter_search()"
```

Rezultati se čuvaju u:
- `results/hyperparameter_search.csv` — tabela svih trial-a
- `results/hpo/` — vizualizacije (parameter importance, parallel coordinate, optimization history)

> **Napomena:** Puni HPO sa 20 trial-a na CPU može trajati 1–2+ sata. Za brži test smanjite `n_trials` u pozivu funkcije.

---

## 6. Rezultati evaluacije

Evaluacija je implementirana u `src/evaluate.py` (notebook `05_evaluation.ipynb`). Na test skupu (4.240 primera) izmerene su sledeće metrike za **LSTM model**:

| Metrika | Vrednost |
|---------|----------|
| Accuracy | 0.2884 |
| Macro F1 | 0.0448 |
| Weighted F1 | 0.1291 |
| Macro AUROC | 0.5280 |

### Detalji po klasama

Classification report pokazuje da model predviđa **isključivo klasu Technical Support** za sve primere:

| Klasa | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Technical Support | 0.29 | 1.00 | 0.45 |
| Ostale klase (9) | 0.00 | 0.00 | 0.00 |

Accuracy od ~0.29 odgovara udelu klase Technical Support u test skupu (~29%), što potvrđuje kolaps modela na većinsku klasu.

### Artefakti evaluacije

Rezultati LSTM modela nalaze se u `results/lstm/`:

- `summary_metrics.csv` — agregirane metrike
- `classification_report.csv` — precision/recall/F1 po klasi
- `confusion_matrix.png` / `.csv` — matrica konfuzije
- `roc_curves.png` / `roc_auc.csv` — ROC krive (one-vs-rest)
- `misclassified_examples.csv` — primeri pogrešno klasifikovanih tiketa

**BERT model** trenutno **nije istreniran** (`models/best_bert.pt` ne postoji), pa evaluacija za BERT nije izvršena.

### Pokretanje evaluacije

```bash
python3 -c "from src.evaluate import evaluate_all; evaluate_all()"
```

---

## 7. Diskusija

### Zašto su rezultati niski?

1. **Kolaps na većinsku klasu.** Model je naučio da uvek predviđa Technical Support (~29% podataka), jer to minimizuje gubitak na nebalansiranom skupu. Accuracy ~0.29 odgovara udelu te klase, dok je macro F1 blizu nule jer ostale klase imaju nulti recall.

2. **Jak nebalans klasa (ratio ~20:1).** Bez tehnika za rešavanje nebalansa (class-weighted loss, oversampling, focal loss) model favorizuje dominantne klase.

3. **LSTM bez pretreniranih embeddinga.** GloVe fajl nije uključen u repozitorijum, pa embedding sloj koristi slučajnu inicijalizaciju. LSTM bez semantički bogatih vektorskih reprezentacija teže uči suptilne razlike između departmana.

4. **Semantičko preklapanje kategorija.** Neke klase imaju sličan vokabular (npr. IT Support vs Technical Support, Product Support vs Customer Service), što otežava diskriminaciju, posebno za jednostavnije modele.

5. **BERT nije istreniran.** Transformer modeli sa pretreniranim jezičkim reprezentacijama tipično postižu znatno bolje rezultate na zadacima klasifikacije teksta u odnosu na LSTM baseline.

### Šta bi pun trening poboljšao?

- **BERT fine-tuning** sa dovoljno epoha i GPU ubrzanjem
- **Primena najboljih hiperparametara** iz Optuna HPO pretrage za finalni LSTM trening
- **Class-weighted cross-entropy** ili focal loss za penalizaciju zanemarivanja manjinskih klasa
- **Preuzimanje GloVe embeddinga** za inicijalizaciju LSTM embedding sloja
- **Duži trening**, data augmentation (parafraziranje, back-translation) ili spajanje semantički sličnih klasa

---

## 8. Zaključak

Projekat implementira kompletan pipeline za klasifikaciju support tiketa: učitavanje i analiza podataka, preprocesiranje, definisanje dva modela (LSTM i BERT), trening sa early stopping-om, hiperparametarska optimizacija pomoću Optuna biblioteke i evaluacija sa detaljnim metrikama i vizualizacijama.

LSTM baseline, iako funkcionalan, pokazuje ograničenja na nebalansiranom skupu od 10 klasa — kolaps na većinsku klasu rezultira niskim macro F1 (~0.045) uprkos umerenoj accuracy (~0.29). Ovo je očekivano ponašanje bez tehnika za rešavanje nebalansa i bez pretreniranih embeddinga.

Sledeći koraci uključuju BERT fine-tuning, punu Optuna pretragu (20 trial-a) i primenu tehnika za nebalansirane klase. Projekat demonstrira praktičnu primenu dubokog učenja za obradu prirodnog jezika u domenu korisničke podrške.

---

## Instalacija

### Preduslovi

- Python 3.10 ili noviji
- pip
- (opciono) CUDA ili Apple Silicon (MPS) za ubrzanje treninga

### Koraci

```bash
git clone https://github.com/<korisnicko-ime>/support-ticket-classifier.git
cd support-ticket-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dataset se automatski preuzima sa HuggingFace-a pri prvom pokretanju pipeline-a. Ako je tokenizer već u lokalnom kešu, možete raditi offline:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## Pokretanje

Preporučeni redosled kroz Jupyter notebookove:

```bash
jupyter notebook notebooks/
```

| Notebook | Sadržaj |
|----------|---------|
| `01_data_exploration.ipynb` | EDA, distribucija klasa, dužina teksta |
| `02_preprocessing.ipynb` | Čišćenje, tokenizacija, split |
| `03_model_architecture.ipynb` | Pregled LSTM i BERT arhitekture |
| `04_training.ipynb` | Trening oba modela |
| `05_evaluation.ipynb` | Evaluacija, confusion matrix, ROC |
| `06_hyperparameter_optimization.ipynb` | Optuna HPO pretraga |

### CLI komande

```bash
# Preprocesiranje (ako processed CSV fajlovi ne postoje)
python3 -c "from src.preprocessing import TextPreprocessor; TextPreprocessor().preprocess_pipeline()"

# Trening LSTM
python3 -c "from src.train import train_lstm; train_lstm()"

# Trening BERT
python3 -c "from src.train import train_bert; train_bert()"

# Hiperparametarska optimizacija
python3 -c "from src.hyperparameter_search import run_hyperparameter_search; run_hyperparameter_search()"

# Evaluacija
python3 -c "from src.evaluate import evaluate_all; evaluate_all()"
```

---

## Struktura projekta

```
support-ticket-classifier/
├── data/
│   ├── processed/              # train.csv, val.csv, test.csv
│   └── raw/                    # tickets.csv (keš, gitignored)
├── logs/
│   └── training/               # istorija treninga (JSON)
├── models/                     # checkpointi (gitignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_architecture.ipynb
│   ├── 04_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_hyperparameter_optimization.ipynb
├── results/
│   ├── lstm/                   # metrike, grafici evaluacije
│   ├── hpo/                    # Optuna vizualizacije
│   └── hyperparameter_search.csv
├── src/
│   ├── __init__.py
│   ├── config.py               # konstante i hiperparametri
│   ├── data_loader.py          # učitavanje HuggingFace dataseta
│   ├── preprocessing.py        # čišćenje, tokenizacija, split
│   ├── model.py                # LSTM i BERT modeli
│   ├── train.py                # trening petlja
│   ├── evaluate.py             # evaluacija i vizualizacije
│   └── hyperparameter_search.py  # Optuna HPO
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Licenca

Ovaj projekat je distribuiran pod **MIT licencom**. Pogledajte [LICENSE](LICENSE) fajl za pun tekst licence.
