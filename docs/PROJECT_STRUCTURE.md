# Project Structure

## Overview

This repository contains a research-oriented machine learning workflow for Alzheimer's disease diagnosis from GEO gene expression datasets.

## Directories

### `ad_diagnosis/`

Core Python package.

- `config.py`: central paths and output locations
- `common.py`: shared data utilities, model definitions, plotting helpers
- `preprocess.py`: build cleaned training matrix and sample labels
- `feature_selection.py`: integrated candidate-gene ranking
- `train_transformer.py`: Transformer training and export
- `train_svm.py`: SVM baselines and ensemble training
- `external_validation.py`: external dataset evaluation
- `plot_gene_expression.py`: candidate-gene expression plots
- `plot_gene_interaction.py`: Transformer interaction analysis

### `scripts/`

Thin command-line wrappers for the main tasks.

- Full pipeline: `python scripts/run_pipeline.py`
- Preprocess only: `python scripts/preprocess_data.py`
- Feature selection only: `python scripts/run_feature_selection.py`
- Transformer only: `python scripts/train_transformer.py`
- SVM only: `python scripts/train_svm.py`
- External validation only: `python scripts/external_validation.py`

### `data/`

Local dataset directory expected by the codebase.

- `GSE33000mx/`: training cohort
- `GSE122063yz1/`: external validation cohort
- `GSE109887yz2/`: external validation cohort

The public project layout assumes these files are managed locally rather than committed with the source code.

### `results/`

Runtime-generated experiment outputs.

The directory is created automatically and typically contains:

- `feature_selection/`: candidate genes and ranking tables
- `transformer/`: checkpoints, parameters, figures, comparison tables
- `svm/`: trained classical models and evaluation outputs
- `external_validation/`: cross-dataset metrics and figures
- `gene_validation/`: candidate-gene expression figures

### `docs/`

Repository documentation, data notes, and experiment summaries.
