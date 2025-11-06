# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains the accompanying code for a research article on microbial growth medium prediction using rule-based and gradient boosting machine learning models. The pipeline processes KG-Microbe knowledge graph data to train binary classifiers predicting whether microbial taxa can grow in specific media (medium 65 and medium 514).

## Data Source

The required KG-Microbe dataset must be downloaded from Zenodo: https://zenodo.org/records/15106978

Download and extract the data to the `src/` directory. The main required file is `merged-kg_edges.tsv`.

## Sequential Pipeline Execution

The analysis follows a strict sequential workflow. Scripts must be run in numerical order:

### 1. Data Preparation (`src/01_prepare_data_binary.py`)

First script to run. Prepares binary classification data from KG-Microbe knowledge graph.

```bash
# Run for a specific medium
python src/01_prepare_data_binary.py --medium 65
python src/01_prepare_data_binary.py --medium 514

# Run for both media
python src/01_prepare_data_binary.py
```

Key functions:
- `run_data_prep(mediumid)`: Main entry point for data preparation
- `clean_singleton_and_constant_features()`: Iteratively removes low-information features/samples
- Extracts subject-object pairs from merged-kg_edges.tsv
- Filters for NCBITaxon/strain subjects and various object types (CHEBI, GO, EC, medium properties, phenotypes)
- Creates one-hot encoded feature matrices
- Generates binary classification targets (growth vs. no-growth in specific media)
- Outputs to `data/taxa_to_media__binary_permute_{mediumid}_data_df_clean.tsv.gz`

### 2. Model Training & Comparison (`src/02_compute_compare_models.py`)

Trains both rule-based (CleverMiner) and CatBoost models.

```bash
python src/02_compute_compare_models.py --medium 514 --model 2
python src/02_compute_compare_models.py --medium 65 --model 0
```

**Note:** The script currently has hardcoded configuration at the top - modify these values directly if not using command-line arguments:
```python
mediumid = 514  # or 65
model = 2       # Valid: 0,1,2 for medium 65; 1,2,3 for medium 514
```

Model configurations per medium:
- **Medium 65:**
  - Model 0: CONFS with robustness constraints (min_base=3, min_additionally_scored=20)
  - Model 1: CONFS without robustness constraints
  - Model 2: CONF quantifier (threshold=0.2)
- **Medium 514:**
  - Model 1: CONFS with strong robustness (min_base=20, min_additionally_scored=50)
  - Model 2: CONFS without robustness constraints
  - Model 3: CONF quantifier (threshold=0.2)

Data split: 70% train, 20% validation, 10% test (stratified, RANDOM_SEED=12)

Outputs classification reports and confusion matrices to `outputs/`

### 3. Feature Importance Agreement (`src/03_compute_feature_importance_agreement.py`)

Analyzes SHAP value correlation between train and test sets to validate feature importance stability.

```bash
python src/03_compute_feature_importance_agreement.py --medium 65
python src/03_compute_feature_importance_agreement.py --medium 514
```

**Note:** The script may have hardcoded configuration at the top:
```python
mediumid = 65  # or 514
```

Key analyses:
- CatBoost with different iteration counts (1, 5, 100)
- SHAP feature importance on train and test data
- Pearson correlation between SHAP values across splits
- Validates feature importance stability across data splits

## Taxonomic Stratification Pipeline

Advanced pipeline with taxonomic stratification located in `src/taxonomic_stratification/`.

### Running Taxonomic Stratification

Use the shell script or run Python directly:

```bash
cd src/taxonomic_stratification
bash run.sh
```

Or individual runs:
```bash
python src/taxonomic_stratification/kg_microbe_train_binary_medium__pipeline.py 65 \
    --data traits \
    --closure false \
    --cv_folds 0 \
    --n_samples 0 \
    --taxonomic-stratify \
    --taxonomic-level family \
    --config src/kg_microbe_train__config_local_rule_mining.json
```

Key parameters:
- First positional arg: medium ID (65 or 514)
- `--data`: Comma-separated data types (traits, ec_rhea, taxonomy)
- `--closure`: Whether to add semantic closure (true/false)
- `--cv_folds`: Number of cross-validation folds (0 = no CV)
- `--n_samples`: Permutation knockoff samples (0 = no permutation)
- `--taxonomic-stratify`: Enable taxonomic stratification
- `--taxonomic-level`: Level for stratification (family, genus, species)
- `--config`: Required JSON config file path

### Utility Modules

**`src/pipeline_utils.py`** - Shared utilities for main pipeline scripts (01-03):
- `load_preprocessed_data()`: Load preprocessed data from step 01
- `create_train_val_test_split()`: Stratified 70/20/10 train/val/test split
- `convert_labels_to_catboost_format()`: Convert labels to CatBoost category indexes
- `train_catboost_model()`: Train CatBoost classifier with validation
- `evaluate_model()`: Generate accuracy, classification report for test set
- `evaluate_model_on_train()`: Check training accuracy (overfitting detection)
- `save_confusion_matrix()`: Create and save confusion matrix heatmap
- Constants: `DEFAULT_RANDOM_SEED=12`, `DEFAULT_CB_SEED=9759`

**`src/taxonomic_stratification/kg_microbe_pipeline_utils.py`** - Advanced pipeline utilities:
- `load_data()`: Load edges and nodes from config paths
- `process_data_pairs()`: Extract and filter subject-object pairs
- `add_closure()`: Add semantic closure to features
- `taxa_media_groups()`: Create one-hot encoded matrices
- `EC_RHEA_annotations()`, `taxonomy_annotations()`, `eggnog_annotations()`, `traithop_annotations()`: Add annotation layers
- `remove_singleton_row_and_col()`: Remove features/samples with single occurrences
- `remove_identical_pattern()`: Deduplicate identical feature columns
- `remove_one_edit_distance_features()`: Remove highly similar binary features
- `remove_low_variance_features()`: Filter low-variance features
- `check_and_split_data()`: Stratified train/val/test split with optional taxonomic stratification
- `perform_permutation_knockoff_analysis()`: Permutation-based feature selection
- `perform_cross_validation()`: K-fold CV with stratification
- `train_final_model()`: Train final CatBoost model
- `prob_calibration()`: Isotonic regression probability calibration
- `final_shap_analysis()`: SHAP-based feature importance with multiple testing correction
- `process_predictions()`: Generate predictions and evaluation metrics

## Key Dependencies

Install all dependencies via:
```bash
pip install -r requirements.txt
```

Critical packages:
- **cleverminer** (1.2.1): Rule-based classification using association rule learning (CONF, CONFS, DBLCONF quantifiers)
- **catboost** (1.2.7): Gradient boosting classifier
- **scikit-learn** (1.6.1): ML utilities, train/test splitting, metrics
- **pandas** (2.2.3), **numpy** (1.26.4): Data manipulation
- **matplotlib** (3.10.1), **seaborn** (0.13.2), **plotly** (6.0.1): Visualization
- **araxai** (0.3.0): Additional utilities

Optional for advanced analysis:
- **shap**: SHAP feature importance analysis (used in script 03)
- **statsmodels**: Multiple testing correction (used in taxonomic stratification pipeline)

## Architecture Notes

### Data Flow
1. Raw KG-Microbe TSV → filtered subject-object pairs
2. Pairs → one-hot encoded binary feature matrix
3. Binary matrix → cleaned (remove singletons, duplicates, low variance)
4. Train/test split with optional taxonomic stratification
5. Model training (CleverMiner rules + CatBoost)
6. SHAP analysis and feature importance validation

### Model Types
- **Rule-based (CleverMiner)**: Uses association rule mining with confidence-based quantifiers (CONF, CONFS, DBLCONF). Produces interpretable if-then rules.
- **CatBoost**: Gradient boosting with categorical feature support. More accurate but less interpretable.

### Stratification Strategy
When `--taxonomic-stratify` is enabled, train/val/test splits maintain taxonomic distribution at the specified level (family/genus/species), preventing data leakage from closely related organisms.

## Output Organization

- `data/`: Processed datasets (TSV.GZ files)
- `outputs/`: Reference run results, classification reports
- `outputs_llm/`: LLM interpretations (not in this repo)

## Development Notes

### Critical Workflow Requirements
- **Sequential execution required:** Must run scripts in numerical order (01 → 02 → 03)
- **Data dependency:** Scripts 02 and 03 expect output from script 01 in `data/` directory
- **Working directory:** All scripts assume working directory is repository root
- **Random seed:** Fixed at `RANDOM_SEED=12` for reproducibility across all splits

### Configuration Approach
- **Main pipeline (scripts 01-03):** Some scripts have hardcoded configuration at the top of the file that may need manual editing if command-line arguments are not fully implemented
- **Taxonomic stratification:** Requires JSON config file (`--config` parameter) specifying all data paths
- **Model selection:** Medium-specific model configurations are embedded in script logic (see model configuration tables above)

### Code Organization
- **Shared utilities:** `src/pipeline_utils.py` is imported by scripts 02 and 03 to avoid code duplication
- **Two parallel pipelines:**
  - Simple pipeline: `src/01_*.py`, `src/02_*.py`, `src/03_*.py` with `src/pipeline_utils.py`
  - Advanced pipeline: `src/taxonomic_stratification/` with its own `kg_microbe_pipeline_utils.py`

### Data Management
- Input data (`merged-kg_edges.tsv`) should be in `src/` directory
- Processed data outputs to `data/` as `.tsv.gz` files
- Model outputs and reports go to `outputs/` directory
- The pipelines are independent - can run simple OR advanced, not both simultaneously
