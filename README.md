# Emotion Detection from Social Media Comments

This project is a clean baseline for classifying English social media comments into 6 emotion classes:

- `joy`
- `sadness`
- `anger`
- `fear`
- `surprise`
- `neutral`

The first milestone focuses on a realistic local pipeline using:

- lightweight text preprocessing
- TF-IDF features
- Logistic Regression
- evaluation with common classification metrics
- a console demo for prediction

## Architecture Overview

The project is split into three simple layers:

- `scripts/` contains terminal entry points such as training and console prediction.
- `src/emotion_detector/` contains reusable project logic: config loading, preprocessing, data loading, training, evaluation, and inference.
- `data/` and `artifacts/` separate input datasets from generated outputs such as trained models and figures.

This keeps the code beginner-readable while still looking like a serious software project instead of a one-file experiment.

## Project Structure

```text
AI/
├─ artifacts/
│  ├─ figures/
│  ├─ metrics/
│  └─ models/
├─ configs/
│  └─ baseline_config.json
├─ data/
│  ├─ external/
│  ├─ processed/
│  └─ raw/
├─ scripts/
│  ├─ evaluate_baseline.py
│  ├─ predict_console.py
│  └─ train_baseline.py
├─ src/
│  └─ emotion_detector/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ data_loader.py
│     ├─ evaluation.py
│     ├─ inference.py
│     ├─ preprocessing.py
│     ├─ training.py
│     └─ utils/
│        ├─ __init__.py
│        └─ io.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```

## What Each Part Does

- `configs/baseline_config.json`: stores dataset path, model settings, output paths, and target labels.
- `data/raw/`: main training dataset CSV files.
- `data/processed/`: optional cleaned or split datasets later.
- `data/external/`: extra public or manually collected data.
- `artifacts/models/`: saved trained sklearn pipeline and label metadata.
- `artifacts/metrics/`: JSON reports with accuracy, precision, recall, and F1-score.
- `artifacts/figures/`: confusion matrix images for reports or slides.
- `src/emotion_detector/preprocessing.py`: lightweight social-media text cleaning.
- `src/emotion_detector/data_loader.py`: dataset loading and validation.
- `src/emotion_detector/training.py`: baseline TF-IDF + Logistic Regression workflow.
- `src/emotion_detector/evaluation.py`: metrics calculation and confusion matrix generation.
- `src/emotion_detector/inference.py`: prediction logic for console use.
- `src/emotion_detector/utils/io.py`: reusable save/load helpers for JSON and joblib.

## Dataset Format

Place your dataset in `data/raw/` as a CSV file with these columns:

- `text`
- `label`

Example:

```csv
text,label
I am so happy today!,joy
This is terrible.,anger
I feel nervous about tomorrow.,fear
That was unexpected!,surprise
```

The label names should stay consistent and lowercase.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train The Baseline Model

1. Put your dataset CSV into `data/raw/`.
2. Update `configs/baseline_config.json` if needed.
3. Run:

```bash
python scripts/train_baseline.py
```

Training outputs will be saved into:

- `artifacts/models/`
- `artifacts/metrics/`
- `artifacts/figures/`

## Evaluate Saved Results

To print the latest saved metrics report:

```bash
python scripts/evaluate_baseline.py
```

## Run Console Inference

After training completes:

```bash
python scripts/predict_console.py
```

Type a comment in the terminal to see:

- predicted emotion
- probability for each emotion class

Type `exit` to close the demo.

## Design Choices

- Preprocessing is intentionally light, because aggressive cleaning can remove emotional signals.
- The sklearn `Pipeline` saves vectorization and classification together, which prevents mismatch during inference.
- The structure is modular so a transformer-based experiment can be added later without rewriting the baseline code.
