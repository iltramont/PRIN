# Structured Information Extraction from Rectal Cancer MRI Reports Using Large Language Models

Master's thesis project — Luca Tramonti

## Overview

This project investigates the use of Large Language Models (LLMs) for automatic extraction of structured clinical data from free-text MRI reports of rectal cancer. The goal is to take an unstructured radiology report and produce a structured JSON object containing all the relevant staging and diagnostic fields, following ESGAR (European Society of Gastrointestinal and Abdominal Radiology) annotation standards.

The extracted fields cover:
- **Primary tumor characteristics**: morphology, clock-face position (hours), wall thickness, cranio-caudal extension, distance from the internal anal orifice
- **Tumor staging**: adipose tissue infiltration, sphincter infiltration, extra-organ infiltration, peritoneal reflection involvement, mesorectal fascia involvement
- **Lymph node assessment**: number of suspicious lymph nodes, lymph node locations, tumor deposits
- **TNM staging conclusions**: T stage, N stage, MRF status, EMVI, metastasis status

Each field falls into one of four task types:
| Type | Fields | Metric |
|------|--------|--------|
| **Regression** | wall thickness, cranio-caudal extension, distance from OAI, lymph node count | MAE, MAPE |
| **Binary classification** | MRF, EMVI, sphincter infiltration, mesorectal fascia involvement, etc. | F1, MCC |
| **Multi-class classification** | morphology, T stage, adipose tissue infiltration, peritoneal reflection | F1 macro, MCC |
| **Multi-label classification** | tumor position, lymph node locations, extra-organ infiltration details | F1 macro, Jaccard |

A custom **weighted composite score** is also computed per report, with domain-specific scoring functions (e.g., partial credit for T3ab vs T3cd, clock-face IoU for hour positions) and consistency penalties for contradictory predictions.

## Models Compared

| Model | Strategy | ID |
|-------|----------|----|
| GPT-4.1 Nano | Zero-shot | `gpt-4.1-nano` |
| GPT-4.1 Nano | Few-shot (random examples) | `gpt-4.1-nano_few-shots` |
| GPT-4.1 Nano | Few-shot (MMR retrieval) | `gpt-4.1-nano_MMR` |
| GPT-4.1 Nano | Fine-tuned | `gpt-4.1-nano_FT` |
| GPT-4.1 Nano | Fine-tuned + oversampling | `gpt-4.1-nano_FT_OS` |
| GPT-4.1 Mini | Zero-shot | `gpt-4.1-mini` |
| GPT-4.1 | Zero-shot | `gpt-4.1` |
| GPT-4.1 | Fine-tuned | `gpt-4.1_FT` |
| GPT-5.4 | Reasoning | `gpt-5.4_reasoning` |
| Mistral Large 3 | Zero-shot | `mistral-large-3` |
| Claude Opus 4.6 | Zero-shot | `opus-4.6` |
| Llama 3.2 3B | Fine-tuned | `llama-3b_FT` |
| Dummy baselines | Most frequent / Uniform / Stratified | `baseline_*` |

## Project Structure

```
PRIN/
├── README.md
├── requirements.txt
├── .gitignore
├── .env                          # API keys (OpenAI, Anthropic, Mistral, HuggingFace)
├── note.txt                      # Annotation rules and data cleaning notes
│
├── src/                          # Python source code
│   ├── constants.py              # Pydantic data model, enums, constants, feature weights
│   ├── model_utils.py            # Field introspection, label encoding, data conversion
│   ├── prompting_utils.py        # System/user prompt generation, few-shot formatting
│   ├── performance_utils.py      # Scoring functions (clock IoU, regression, custom T/N scoring)
│   ├── pulisci_dati.py           # Data cleaning pipeline (from raw CSV to clean CSV)
│   ├── splitta_dati.py           # Stratified train/validation/test split
│   ├── dummy_predictions.py      # Dummy baseline predictions
│   ├── analyse_results.py        # Per-field metrics computation (F1, MCC, MAE, etc.)
│   ├── performance.py            # Aggregate weighted score computation across models
│   └── plot_results.py           # Confusion matrices and regression plots per model
│
├── notebooks/                    # Jupyter notebooks
│   ├── EDA.ipynb                 # Exploratory Data Analysis on raw data
│   ├── EDA_results.ipynb         # Analysis of inference results
│   ├── EDA_tesi.ipynb            # EDA figures for the thesis document
│   ├── openai_inference.ipynb    # GPT-4.1 inference pipeline
│   ├── openai_inference_examples.ipynb  # GPT-4.1 with few-shot examples
│   ├── openai_training.ipynb     # OpenAI fine-tuning pipeline
│   ├── anthropic_inference.ipynb # Claude Opus inference pipeline
│   ├── anthropic_inference_examples.ipynb  # Claude with few-shot examples
│   ├── mistral_inference.ipynb   # Mistral Large inference pipeline
│   ├── mistral_training.ipynb    # Mistral fine-tuning pipeline
│   ├── mistral_notebook.ipynb    # Mistral experiments
│   ├── huggingface_inference.ipynb  # Llama inference via HuggingFace
│   ├── huggingface_training.ipynb   # Llama fine-tuning via HuggingFace
│   ├── embeddings.ipynb          # Mistral embeddings for MMR retrieval
│   ├── prompting_guides.ipynb    # Prompt engineering experiments
│   ├── confronto.ipynb           # Model comparison analysis
│   ├── model_comparison.ipynb    # Cross-model comparison and visualization
│   ├── colab.ipynb               # Google Colab-compatible notebook
│   └── prove.ipynb               # Miscellaneous experiments
│
├── data/
│   ├── base.tumoreprimitivo_febbraio.csv  # Raw annotated dataset
│   ├── tumoreprimitivo_clean.csv          # Cleaned dataset
│   ├── train_split.csv                     # Training split
│   ├── validation_split.csv                # Validation split
│   ├── test_split.csv                      # Test split
│   ├── prompts/                            # System prompt templates
│   ├── ft-dataset/                         # Fine-tuning datasets (OpenAI, Mistral JSONL)
│   ├── embeddings/                         # Mistral embeddings for MMR retrieval
│   ├── inference/                          # Model predictions (JSONL per model)
│   └── metrics/                            # Computed metrics (CSV per model)
│
└── figures/                      # Output plots and figures
    ├── *.pdf                     # Distribution plots, comparison heatmaps
    └── <model_name>/             # Per-model confusion matrices and regression plots
```

## Pipeline

The project follows a sequential pipeline:

1. **Data cleaning** (`src/pulisci_dati.py`): Loads raw annotated CSV, removes unclear/duplicate reports, applies domain-specific corrections (staging rules, aggregation of categories), and produces a clean dataset.

2. **Data splitting** (`src/splitta_dati.py`): Performs stratified iterative train/validation/test splitting (60/20/20) using `skmultilearn` to preserve class distributions across multi-label fields.

3. **Inference** (`notebooks/openai_*.ipynb`, `notebooks/anthropic_*.ipynb`, `notebooks/mistral_*.ipynb`, `notebooks/huggingface_*.ipynb`): Sends reports to each LLM with a structured system prompt containing the Pydantic JSON schema. Supports zero-shot, few-shot (random and MMR-based example retrieval), and fine-tuned models.

4. **Baseline generation** (`src/dummy_predictions.py`): Creates dummy predictions using scikit-learn's `DummyClassifier` and `DummyRegressor` with most-frequent, uniform, and stratified strategies.

5. **Metrics computation** (`src/analyse_results.py`): Computes per-field metrics (F1, MCC, MAE, MAPE, Jaccard) across train/validation/test splits for each model.

6. **Aggregate scoring** (`src/performance.py`): Computes a weighted composite score per report using domain-specific scoring functions and consistency penalties.

7. **Visualization** (`src/plot_results.py`, `notebooks/model_comparison.ipynb`): Generates confusion matrices, residual plots, QQ-plots, boxplots, and heatmaps for thesis and paper figures.

## Data Model

The core data model is defined in `src/constants.py` as a Pydantic `BaseModel` (`RectalCancerStagingData`). Each field uses a typed enum to constrain valid values:

```python
class RectalCancerStagingData(BaseModel):
    morfologia: Morfologia               # solido_polipoide | solido_anulare | mucinoso
    ore_inizio: Optional[int]             # Clock-face start position (0-12)
    ore_fine: Optional[int]               # Clock-face end position (0-12)
    spessore_parietale: Optional[int]     # Wall thickness (mm)
    estensione_cranio_caudale: Optional[int]  # Cranio-caudal extension (mm)
    distanza_oai: Optional[int]           # Distance from internal anal orifice (mm)
    posizione: PosizioneFlags             # Multi-label: basso, medio, alto, giunzione
    # ... 15 more fields covering staging, lymph nodes, and conclusions
```

LLMs receive the schema as part of the system prompt and must produce a JSON object conforming to it.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file with:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
MISTRAL_API_KEY=...
HF_TOKEN=...
```

## Running

Scripts are designed to be run from the `src/` directory:

```bash
cd src
python pulisci_dati.py        # Step 1: Clean raw data
python splitta_dati.py        # Step 2: Split into train/val/test
python dummy_predictions.py   # Step 4: Generate baselines
python analyse_results.py     # Step 5: Compute metrics
python performance.py         # Step 6: Aggregate scores
python plot_results.py        # Step 7: Generate plots
```

Inference (Step 3) is performed through the Jupyter notebooks in `notebooks/`.
