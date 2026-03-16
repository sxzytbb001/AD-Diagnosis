import json
import os

import torch

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PACKAGE_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'GSE33000mx')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

FEATURE_SELECTION_DIR = os.path.join(RESULTS_DIR, 'feature_selection')
TRANSFORMER_DIR = os.path.join(RESULTS_DIR, 'transformer')
SVM_DIR = os.path.join(RESULTS_DIR, 'svm')
EXTERNAL_VALIDATION_DIR = os.path.join(RESULTS_DIR, 'external_validation')
GENE_VALIDATION_DIR = os.path.join(RESULTS_DIR, 'gene_validation')

CANDIDATE_GENES_PATH = os.path.join(FEATURE_SELECTION_DIR, 'candidate_genes.txt')
FEATURE_STATS_PATH = os.path.join(FEATURE_SELECTION_DIR, 'candidate_gene_stats.csv')

TRANSFORMER_MODEL_PATH = os.path.join(TRANSFORMER_DIR, 'best_transformer_model.pth')
TRANSFORMER_PARAMS_PATH = os.path.join(TRANSFORMER_DIR, 'best_params.json')
TRANSFORMER_SCALER_PATH = os.path.join(TRANSFORMER_DIR, 'scaler.pkl')

SVM_MODEL_PATH = os.path.join(SVM_DIR, 'best_svm_model.pkl')
SVM_VOTING_PATH = os.path.join(SVM_DIR, 'voting_svm_model.pkl')
SVM_BAGGING_PATH = os.path.join(SVM_DIR, 'bagging_svm_model.pkl')
SVM_ENSEMBLE_PATH = os.path.join(SVM_DIR, 'ensemble_svm_model.pkl')
SVM_SCALER_PATH = os.path.join(SVM_DIR, 'scaler.pkl')
SVM_PARAMS_PATH = os.path.join(SVM_DIR, 'best_params.json')

# 向后兼容别名
MODEL_PATH = TRANSFORMER_MODEL_PATH
BEST_PARAMS_PATH = TRANSFORMER_PARAMS_PATH

TRAIN_MATRIX_PATH = os.path.join(TRAIN_DATA_DIR, 'cleaned_gene_matrix.csv')
TRAIN_LABELS_PATH = os.path.join(TRAIN_DATA_DIR, 'sample_labels.csv')

EXTERNAL_DATA_DIRS = {
    'GSE122063': os.path.join(DATA_DIR, 'GSE122063yz1'),
    'GSE109887': os.path.join(DATA_DIR, 'GSE109887yz2')
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_dirs():
    dirs = [
        RESULTS_DIR,
        FEATURE_SELECTION_DIR,
        TRANSFORMER_DIR,
        SVM_DIR,
        EXTERNAL_VALIDATION_DIR,
        GENE_VALIDATION_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_candidate_genes():
    if os.path.exists(CANDIDATE_GENES_PATH):
        with open(CANDIDATE_GENES_PATH, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return []


def load_best_params():
    if os.path.exists(TRANSFORMER_PARAMS_PATH):
        with open(TRANSFORMER_PARAMS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.3,
        'lr': 0.001,
        'batch_size': 32,
        'weight_decay': 0.01,
        'max_epochs': 80
    }
