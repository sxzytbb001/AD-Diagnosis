# AD-Insight: Alzheimer's Diagnosis with Integrated Feature Selection and Transformer-SVM Comparison

Language: **English** | [简体中文](README_CN.md)

About: 基于集成特征选择与 Transformer 对比 SVM 的阿尔茨海默症诊断研究 / Alzheimer's diagnosis study based on integrated feature selection and Transformer-vs-SVM comparison.

Short name: **AD-Insight**

Research framework for Alzheimer's disease diagnosis from GEO gene expression data.

This project provides a reproducible workflow for:

1. preprocessing a training cohort,
2. selecting candidate genes with an integrated ranking strategy,
3. training a lightweight Transformer classifier,
4. training SVM baselines,
5. evaluating both model families on external cohorts.

## Features

- End-to-end workflow from raw matrix preprocessing to external validation.
- Shared candidate-gene set for fair Transformer vs. SVM comparison.
- Cross-platform validation on independent GEO datasets.
- Optional interpretation scripts for gene expression and gene interaction analysis.
- Runtime-generated output structure for experiments and figures.

## Repository Structure

```text
.
|-- ad_diagnosis/              # Core Python package
|-- scripts/                   # Command-line entry scripts
|-- docs/                      # Documentation and experiment notes
|-- requirements.txt
|-- README.md
`-- README_CN.md
```

Public release notes:

- Raw data is expected under `data/` locally and is not bundled as part of the publish-ready repository layout.
- Generated outputs are written to `results/` at runtime and are not versioned in Git.

## Requirements

- Python 3.10+
- `pip`
- Optional CUDA-enabled PyTorch environment for GPU training

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Layout

Source website: NCBI Gene Expression Omnibus (GEO).

Dataset accession pages:

- `GSE33000`: `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33000`
- `GSE122063`: `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE122063`
- `GSE109887`: `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109887`

Download the required files from GEO and place them under a local `data/` directory.

Create a local `data/` directory with the following structure:

```text
data/
|-- GSE33000mx/
|-- GSE122063yz1/
`-- GSE109887yz2/
```

Expected training files:

- `data/GSE33000mx/geneMatrix.txt`
- `data/GSE33000mx/clinical.xlsx`

Expected external-validation files:

- `geneMatrix.txt`
- `s1.txt`
- `s2.txt`

Notes:

- The repository uses local folder aliases `GSE33000mx`, `GSE122063yz1`, and `GSE109887yz2` for historical compatibility.
- If the downloaded filenames differ from the names above, rename or preprocess them to match the expected layout before running the pipeline.

After preprocessing, the training cohort will also contain:

- `data/GSE33000mx/cleaned_gene_matrix.csv`
- `data/GSE33000mx/sample_labels.csv`

More detail: `docs/DATA_NOTICE.md`

## Quick Start

Run the full pipeline:

```bash
python scripts/run_pipeline.py
```

Run the full pipeline with optional analyses:

```bash
python scripts/run_pipeline.py --with-gene-expression --with-gene-interaction
```

Run each stage separately:

```bash
python scripts/preprocess_data.py
python scripts/run_feature_selection.py
python scripts/train_transformer.py
python scripts/train_svm.py
python scripts/external_validation.py
python scripts/plot_gene_expression.py
python scripts/plot_gene_interaction.py
```

## Pipeline Stages

### 1. Preprocessing

Builds the cleaned training matrix and binary sample labels from the training cohort metadata.

### 2. Feature Selection

Combines seven ranking strategies:

1. T-test
2. Mutual Information
3. XGBoost importance
4. Random Forest importance
5. Elastic Net
6. mRMR
7. Stability Selection

Outputs are written to `results/feature_selection/`.

### 3. Transformer Training

Trains a lightweight Transformer-based classifier with multi-seed evaluation and saves model artifacts to `results/transformer/`.

### 4. SVM Training

Trains classical baselines, including single SVM, voting SVM, and bagging SVM, and saves artifacts to `results/svm/`.

### 5. External Validation

Evaluates saved models on external cohorts and writes metrics and figures to `results/external_validation/`.

### 6. Optional Interpretation

- `scripts/plot_gene_expression.py`
- `scripts/plot_gene_interaction.py`

These scripts generate thesis-style interpretation figures under `results/gene_validation/` and `results/transformer/`.

## Generated Outputs

The `results/` directory is created automatically on first run with this structure:

```text
results/
|-- feature_selection/
|-- transformer/
|-- svm/
|-- external_validation/
`-- gene_validation/
```

Typical outputs include:

- candidate gene rankings,
- saved model parameters and preprocessors,
- ROC curves and confusion matrices,
- external validation summary tables,
- gene-importance and interaction visualizations.

## Documentation

- Chinese overview: `README_CN.md`
- Project structure: `docs/PROJECT_STRUCTURE.md`
- Data setup notes: `docs/DATA_NOTICE.md`
- Historical experiment summary: `docs/EXPERIMENT_SUMMARY.md`

## Notes

- This is a research codebase intended for experimentation and thesis-style analysis.
- There is no automated test suite or CI workflow in the current release.
- Chinese plots use the `SimHei` font if available; rendering may vary by environment.
