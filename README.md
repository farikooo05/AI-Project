# Emotion Detection from Social Media Comments

This project classifies English text comments into 6 emotion labels:

- `joy`
- `sadness`
- `anger`
- `fear`
- `surprise`
- `neutral`

The repository contains two parallel tracks:

- a local baseline pipeline built with TF-IDF + Logistic Regression
- a transformer workflow where data preparation and local inference live here, while GPU training is typically done in Google Colab

The project is designed to stay beginner-readable while still being modular enough for a university report, demo, and later experimentation.

## If You Are New To This Repository

You do not need to understand the whole codebase at once.

Start with these ideas:

- the main cleaned dataset is `data/processed/goemotions_6class.csv`
- the baseline model is the simple local reference model
- the transformer model is the stronger model, usually trained in Google Colab
- `scripts/` contains the commands you actually run
- `artifacts/` contains saved outputs such as models, metrics, figures, and reports

If you are taking over the project from another teammate, the safest reading order is:

1. this README
2. `configs/goemotions_label_mapping.json`
3. `scripts/run_goemotions_pipeline.py`
4. `scripts/train_baseline.py`
5. `scripts/predict_console.py`
6. `scripts/predict_transformer.py`

## Final Project Structure

```text
AI/
|- archive/
|  |- legacy_configs/
|  |- legacy_data/
|  `- legacy_modules/
|- artifacts/
|  |- figures/
|  |- metrics/
|  |- models/
|  `- reports/
|- configs/
|- data/
|  |- external/
|  |- processed/
|  `- raw/
|- scripts/
|- src/
|  `- emotion_detector/
|- .gitignore
|- README.md
`- requirements.txt
```

## Main Directories

- `src/emotion_detector/`: reusable project logic
- `scripts/`: terminal entry points for the main workflows
- `configs/`: JSON configuration files, including the GoEmotions mapping
- `data/raw/`: downloaded raw dataset files
- `data/processed/`: cleaned and merged training CSV files
- `artifacts/models/`: saved baseline and transformer artifacts
- `artifacts/metrics/`: baseline metrics and prediction analysis outputs
- `artifacts/reports/`: dataset audits, augmentation reports, merge reports, and transformer training reports
- `archive/`: conservatively archived legacy files that are no longer part of the recommended workflow

## What Is Already Prepared

At this stage of the project, the repository already contains:

- a cleaned GoEmotions-based 6-class dataset
- a trained baseline model and saved baseline metrics
- baseline error-analysis files
- a curated additional dataset for augmentation experiments
- a merged dataset for transformer retraining experiments
- active local transformer inference artifacts

This means the project is already in a usable state for report writing, presentation preparation, and final polishing.

## Recommended Workflow

### 1. Build the cleaned GoEmotions dataset

```bash
python scripts/run_goemotions_pipeline.py
```

This does two things:

- downloads the official GoEmotions CSV parts into `data/raw/goemotions_source/`
- converts and maps them into the final 6-label dataset at `data/processed/goemotions_6class.csv`

Important mapping file:

- `configs/goemotions_label_mapping.json`

Dataset preparation reports:

- `artifacts/reports/dataset_preparation/mapping_audit.json`
- `artifacts/reports/dataset_preparation/class_distribution.csv`
- `artifacts/reports/dataset_preparation/class_weights.csv`
- `artifacts/reports/dataset_preparation/sample_examples_per_class.txt`
- `artifacts/reports/dataset_preparation/focused_examples_per_class.txt`
- `artifacts/reports/dataset_preparation/source_label_groups.json`
- `artifacts/reports/dataset_preparation/warnings.txt`

### 2. Train the local baseline

```bash
python scripts/train_baseline.py
python scripts/evaluate_baseline.py
python scripts/analyze_predictions.py
```

Baseline outputs:

- model: `artifacts/models/baseline_pipeline.joblib`
- labels: `artifacts/models/labels.json`
- metrics: `artifacts/metrics/baseline_metrics.json`
- confusion matrix image: `artifacts/figures/confusion_matrix.png`
- prediction analysis: `artifacts/metrics/prediction_analysis/`

### 3. Run the baseline console demo

```bash
python scripts/predict_console.py
```

The demo:

- loops until the user types `exit`, `quit`, or `q`
- shows the top 3 predicted classes
- prints compact probability output
- shows lightweight feature-level explainability for the TF-IDF + Logistic Regression baseline

### 4. Optional stricter mapping experiments

You can temporarily exclude broad source labels without editing the main mapping file:

```bash
python scripts/prepare_dataset.py --exclude-source-labels gratitude relief
```

This is useful for experiments such as testing whether `gratitude` or `relief` make `joy` too broad.

### 5. Optional curated additional dataset

Generate a separate high-precision dataset intended to strengthen class boundaries:

```bash
python scripts/create_additional_emotion_dataset.py
```

Outputs:

- dataset: `data/processed/emotion_additional_curated.csv`
- summary: `artifacts/reports/additional_dataset/summary.json`
- examples: `artifacts/reports/additional_dataset/examples_per_class.txt`
- label guidelines: `artifacts/reports/additional_dataset/label_guidelines.json`

This dataset is additive. It is not a replacement for the cleaned GoEmotions dataset.

### 6. Optional merged dataset for transformer retraining

```bash
python scripts/merge_and_train_transformer.py
```

This workflow:

- loads `data/processed/goemotions_6class.csv`
- loads `data/processed/emotion_additional_curated.csv`
- adds a `data_source` column
- concatenates and shuffles the data
- saves `data/processed/emotion_training_merged.csv`
- retrains a transformer and saves comparison reports

Merge reports:

- `artifacts/reports/merged_dataset/merge_summary.json`
- `artifacts/reports/merged_dataset/label_distribution.csv`
- `artifacts/reports/merged_dataset/data_source_distribution.csv`
- `artifacts/reports/merged_dataset/source_label_distribution.csv`

### 7. Train the transformer

The repository includes a local/Colab-friendly training entry point:

```bash
python scripts/train_transformer.py
```

Common options:

```bash
python scripts/train_transformer.py --epochs 4 --learning-rate 2e-5
python scripts/train_transformer.py --no-class-weights
```

Typical usage:

- prepare the dataset locally
- upload the processed CSV to Google Colab
- fine-tune the transformer with GPU
- download the exported model artifacts
- copy them back into `artifacts/models/transformer/`

Transformer training reports:

- `artifacts/reports/transformer_training/class_weights.json`
- `artifacts/reports/transformer_training/evaluation_report.json`
- `artifacts/reports/transformer_training/confusion_matrix.csv`
- `artifacts/reports/transformer_training/misclassified_examples.csv`
- `artifacts/reports/transformer_training/top_confusions.json`

The per-class precision, recall, and F1 values are stored inside
`evaluation_report.json` under the saved classification report.

## Transformer Artifact Workflow

The transformer workflow is intentionally separated from the baseline.

1. Prepare the dataset locally in this repository.
2. Train the transformer in Google Colab or another GPU-backed environment.
3. Export the model artifacts.
4. Copy them into `artifacts/models/transformer/`.
5. Run local transformer inference from the saved artifacts.

Expected artifact files:

- `config.json`
- `label_mappings.json`
- `model.safetensors` or `pytorch_model.bin`
- `tokenizer.json` or `vocab.txt`
- `tokenizer_config.json`
- `special_tokens_map.json`

The local loader validates the label mapping contract between `config.json` and `label_mappings.json` so mismatched artifacts fail clearly instead of silently.

Backup transformer folders are kept under `artifacts/models/transformer_backup_*` when older downloaded artifacts are preserved.

### Local transformer inference

```bash
python scripts/predict_transformer.py --text "I am so happy today!"
```

### Diagnostic transformer tests

```bash
python scripts/test_transformer_examples.py
```

This script runs a reusable set of easy, medium, and hard diagnostic sentences so you can compare transformer behavior across retraining runs.

## What To Rerun And What Not To Rerun

You do not need to rerun every step every time.

Rerun the dataset pipeline only if:

- you changed the GoEmotions label mapping
- you want a stricter dataset variant
- you want to rebuild the processed CSV from scratch

Rerun the baseline only if:

- the processed training dataset changed
- you want updated baseline metrics or fresh error-analysis files

Rerun the transformer only if:

- you changed the training dataset
- you want a stronger final model
- you are testing the merged dataset or another training variation

If the current artifacts already match your final chosen dataset and model version, you can stop training and move on to the report and presentation.

## Label Mapping Strategy

The project uses a stricter GoEmotions-to-6-class mapping than the original early version.

Key rule:

- if a source label does not fit clearly into one of the final 6 classes, it is dropped instead of forced into a noisy bucket

Examples of dropped labels:

- `admiration`
- `approval`
- `caring`
- `confusion`
- `curiosity`
- `desire`
- `embarrassment`
- `love`
- `optimism`
- `pride`
- `realization`

This makes the final dataset smaller, but usually cleaner and more reliable.

## Imbalance Handling

The project handles imbalance in two practical ways:

- the baseline uses `class_weight="balanced"` in Logistic Regression
- the transformer training workflow supports weighted cross-entropy based on train-split class weights

The goal is not to overengineer balancing, but to keep minority classes such as `fear` and `surprise` from being ignored.

## Important Scripts

- `scripts/run_goemotions_pipeline.py`: download raw GoEmotions and build the cleaned 6-class dataset
- `scripts/prepare_dataset.py`: manually run dataset preparation and stricter mapping experiments
- `scripts/train_baseline.py`: train the TF-IDF + Logistic Regression baseline
- `scripts/evaluate_baseline.py`: print the latest saved baseline metrics
- `scripts/analyze_predictions.py`: generate baseline prediction analysis artifacts
- `scripts/predict_console.py`: run the local baseline demo
- `scripts/create_additional_emotion_dataset.py`: generate the curated augmentation dataset
- `scripts/train_transformer.py`: train the transformer on a prepared CSV
- `scripts/merge_and_train_transformer.py`: merge the main and curated datasets, then retrain the transformer
- `scripts/predict_transformer.py`: run local transformer inference from exported artifacts
- `scripts/test_transformer_examples.py`: run reusable diagnostic test sentences against the current transformer

## Important Data Files

- main cleaned dataset: `data/processed/goemotions_6class.csv`
- stricter experiment variant: `data/processed/goemotions_6class_strict_joy.csv`
- curated additional dataset: `data/processed/emotion_additional_curated.csv`
- merged transformer dataset: `data/processed/emotion_training_merged.csv`

## Limitations

- The final 6-label setup is still coarser than real emotional language.
- Some comments naturally contain mixed emotions, but the current project stays single-label.
- Transformer training is best done in Colab because local CPU training is slow.
- Label quality is still a bigger risk than model choice when mapping many original emotions into a small final taxonomy.

## Current Recommended Submission/Demo Flow

For the final baseline:

```bash
python scripts/run_goemotions_pipeline.py
python scripts/train_baseline.py
python scripts/evaluate_baseline.py
python scripts/analyze_predictions.py
python scripts/predict_console.py
```

For the final transformer:

1. prepare or merge the dataset locally
2. train in Colab with GPU
3. copy exported artifacts into `artifacts/models/transformer/`
4. test locally with:

```bash
python scripts/predict_transformer.py --text "That scared me."
python scripts/test_transformer_examples.py
```

## Handoff Notes For Teammates

If you are continuing this project from another teammate:

- use the processed datasets under `data/processed/`
- treat `artifacts/models/transformer/` as the active local transformer model
- keep the backup transformer folders unless you are sure you no longer need rollback
- use `artifacts/reports/` when writing the report, because that folder contains the evidence for dataset cleaning, augmentation, and merge experiments
- use `artifacts/metrics/` and `artifacts/figures/` for baseline results and baseline error analysis

If you only need to finish the report and presentation, you likely do not need to retrain anything unless you intentionally want to update the final results.

## Archived Legacy Items

The repository keeps a small `archive/` folder for conservative cleanup. Items moved there are not part of the recommended workflow anymore, but they were preserved instead of deleted in case you still want to inspect old experiments or earlier helper files.
