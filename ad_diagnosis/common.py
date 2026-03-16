import os
import json
import copy
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from sklearn.preprocessing import QuantileTransformer, StandardScaler, quantile_transform
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from . import config

config.ensure_dirs()
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def tensor_to_numpy(tensor):
    """在 Torch 与 NumPy 二进制不兼容时，避免直接调用 tensor.numpy()。"""
    if isinstance(tensor, torch.Tensor):
        return np.asarray(tensor.detach().cpu().tolist())
    return np.asarray(tensor)


def load_training_matrix_and_labels():
    if not os.path.exists(config.TRAIN_MATRIX_PATH):
        raise FileNotFoundError(f'找不到训练矩阵文件: {config.TRAIN_MATRIX_PATH}')
    if not os.path.exists(config.TRAIN_LABELS_PATH):
        raise FileNotFoundError(f'找不到训练标签文件: {config.TRAIN_LABELS_PATH}')

    gene_matrix = pd.read_csv(config.TRAIN_MATRIX_PATH, index_col=0).T
    labels_df = pd.read_csv(config.TRAIN_LABELS_PATH)

    if not all(gene_matrix.index == labels_df['sample_id']):
        gene_matrix = gene_matrix.loc[labels_df['sample_id']]

    y = (labels_df['label'] == 'AD').astype(int).values
    return gene_matrix, labels_df, y


def load_training_data(candidate_genes):
    gene_matrix, labels_df, y = load_training_matrix_and_labels()
    available_genes = [g for g in candidate_genes if g in gene_matrix.columns]
    X = gene_matrix[available_genes].values
    return X, y, available_genes, gene_matrix, labels_df


def load_external_data(data_dir, dataset_name, candidate_genes):
    matrix_path = os.path.join(data_dir, 'geneMatrix.txt')
    s1_path = os.path.join(data_dir, 's1.txt')
    s2_path = os.path.join(data_dir, 's2.txt')

    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f'{dataset_name} 缺少文件: {matrix_path}')
    if not os.path.exists(s1_path) or not os.path.exists(s2_path):
        raise FileNotFoundError(f'{dataset_name} 缺少 s1/s2 分组文件')

    try:
        df = pd.read_csv(matrix_path, sep='\t', index_col=0)
    except Exception:
        df = pd.read_csv(matrix_path, sep='\s+', index_col=0)

    with open(s1_path, 'r', encoding='utf-8') as f:
        s1 = [x.strip() for x in f if x.strip()]
    with open(s2_path, 'r', encoding='utf-8') as f:
        s2 = [x.strip() for x in f if x.strip()]

    valid_s1 = [x for x in s1 if x in df.columns]
    valid_s2 = [x for x in s2 if x in df.columns]

    X = pd.concat([df[valid_s1].T, df[valid_s2].T], axis=0)
    y = np.array([0] * len(valid_s1) + [1] * len(valid_s2))

    X_final = pd.DataFrame(index=X.index)
    for gene in candidate_genes:
        X_final[gene] = X[gene] if gene in X.columns else 0.0

    return X_final.values, y, {'control': len(valid_s1), 'ad': len(valid_s2)}


def align_dataset_labels(X_train, y_train, X_val, y_val):
    train_sig = np.mean(X_train[y_train == 1], axis=0) - np.mean(X_train[y_train == 0], axis=0)
    val_sig = np.mean(X_val[y_val == 1], axis=0) - np.mean(X_val[y_val == 0], axis=0)

    valid_idx = np.where((np.abs(train_sig) > 1e-9) | (np.abs(val_sig) > 1e-9))[0]
    if len(valid_idx) < 3:
        return y_val, np.nan, False

    corr = np.corrcoef(train_sig[valid_idx], val_sig[valid_idx])[0, 1]
    if np.isnan(corr):
        return y_val, corr, False
    if corr < 0:
        return 1 - y_val, corr, True
    return y_val, corr, False


def align_data_distribution(X_train, X_ext):
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0) + 1e-8
    ext_mean = X_ext.mean(axis=0)
    ext_std = X_ext.std(axis=0) + 1e-8

    X_ext_aligned = (X_ext - ext_mean) / ext_std
    X_ext_aligned = X_ext_aligned * train_std + train_mean
    return np.clip(X_ext_aligned, -2, 2)


def align_data_distribution_soft(X_train, X_ext, alpha=0.5):
    """温和的分布对齐：保留部分原始数据结构，适用于 Transformer 等非线性模型。

    alpha=1.0 等同于全量对齐（传统方法），alpha=0.0 不做任何对齐。
    默认 alpha=0.5 在对齐和保留原始非线性结构之间取折中。
    """
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0) + 1e-8
    ext_mean = X_ext.mean(axis=0)
    ext_std = X_ext.std(axis=0) + 1e-8

    # 全量对齐版本
    X_full_aligned = (X_ext - ext_mean) / ext_std * train_std + train_mean
    # 仅做中心对齐（保留相对结构）
    X_center_aligned = X_ext - ext_mean + train_mean

    # 混合：alpha 控制对齐程度
    X_blended = alpha * X_full_aligned + (1 - alpha) * X_center_aligned
    return np.clip(X_blended, -3, 3)


def quantile_normalize_external(X):
    X_qt = quantile_transform(
        X,
        n_quantiles=min(100, X.shape[0]),
        output_distribution='normal',
        copy=True
    )
    return np.clip(X_qt, -2, 2)


def fit_rank_gauss_preprocessor(X, max_quantiles=128):
    n_quantiles = max(16, min(int(max_quantiles), int(X.shape[0])))
    quantile = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution='normal',
        copy=True,
        subsample=int(1e9),
        random_state=42,
    )
    scaler = StandardScaler()
    X_rank = quantile.fit_transform(X)
    X_scaled = scaler.fit_transform(X_rank)
    return {
        'type': 'rank_gauss_standard',
        'quantile': quantile,
        'scaler': scaler,
    }, np.clip(X_scaled, -4, 4)


def transform_with_preprocessor(preprocessor, X):
    if type(preprocessor) is dict:
        if preprocessor.get('type') == 'rank_gauss_standard':
            X_rank = preprocessor['quantile'].transform(X)
            X_scaled = preprocessor['scaler'].transform(X_rank)
            return np.clip(X_scaled, -4, 4)
        raise TypeError('Unsupported preprocessor mapping for transform.')
    if hasattr(preprocessor, 'transform'):
        return preprocessor.transform(X)
    raise TypeError('Unsupported preprocessor type for transform.')


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty(n, dtype=float)
    cumulative = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        cumulative = min(cumulative, value)
        adjusted[i] = cumulative
    result = np.empty(n, dtype=float)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def cohens_d(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n1 = len(x1)
    n2 = len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1 = np.var(x1, ddof=1)
    s2 = np.var(x2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(n1 + n2 - 2, 1) + 1e-8)
    return float((np.mean(x1) - np.mean(x2)) / pooled)


def create_weighted_sampler(y):
    class_counts = np.bincount(y.astype(int))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y.astype(int)].tolist()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.best_state = None
        self.counter = 0

    def step(self, score, model):
        if self.best_score is None:
            improved = True
        elif self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=return_attention,
            average_attn_weights=False
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, attn_weights if return_attention else None


class TransformerV3(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.input_dim = input_dim
        self.raw_proj = nn.Linear(input_dim, d_model)
        self.interaction_factors = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.value_proj = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.gene_embedding = nn.Parameter(torch.randn(1, input_dim + 1, d_model) * 0.02)
        self.gene_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )

    def forward(self, x, return_attention=False, return_features=False, return_gene_weights=False):
        raw_x = x
        x = x.unsqueeze(-1)
        x = self.value_proj(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.input_norm(x + self.gene_embedding[:, :x.size(1), :])
        x = self.input_dropout(x)

        attention_maps = []
        for layer in self.encoder_layers:
            x, attn = layer(x, return_attention=return_attention)
            if return_attention and attn is not None:
                attention_maps.append(attn)

        cls_feature = x[:, 0]
        gene_tokens = x[:, 1:]
        gate_logits = self.gene_gate(gene_tokens).squeeze(-1)
        gate_weights = torch.softmax(gate_logits, dim=1)
        pooled_feature = torch.sum(gene_tokens * gate_weights.unsqueeze(-1), dim=1)
        linear_feature = self.raw_proj(raw_x)
        factor_input = raw_x.unsqueeze(-1) * self.interaction_factors.unsqueeze(0)
        interaction_feature = 0.5 * ((factor_input.sum(dim=1) ** 2) - (factor_input ** 2).sum(dim=1))
        features = (
            0.50 * cls_feature
            + 0.20 * pooled_feature
            + 0.15 * linear_feature
            + 0.15 * interaction_feature
        )
        logits = self.classifier(features)

        outputs = [logits]
        if return_features:
            outputs.append(features)
        if return_attention:
            outputs.append(attention_maps)
        if return_gene_weights:
            outputs.append(gate_weights)
        if len(outputs) == 1:
            return logits
        return tuple(outputs)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def plot_probability_distribution(y_true, y_prob, threshold, save_path, title):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_prob[y_true == 0], fill=True, alpha=0.3, color='#4c72b0', label='Control', warn_singular=False)
    sns.kdeplot(y_prob[y_true == 1], fill=True, alpha=0.3, color='#dd8452', label='AD', warn_singular=False)
    plt.axvline(threshold, color='green', linestyle='--', label=f'阈值={threshold:.2f}')
    plt.title(title)
    plt.xlabel('AD 概率')
    plt.ylabel('密度')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tsne(features, y_true, save_path, title):
    try:
        perplexity = min(20, max(5, len(y_true) - 1))
        embedded = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            init='random',
            learning_rate='auto'
        ).fit_transform(features)

        plt.figure(figsize=(7, 7))
        scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=y_true, cmap='coolwarm', alpha=0.8, edgecolors='k')
        handles, _ = scatter.legend_elements()
        plt.legend(handles, ['Control', 'AD'], title='Group')
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception:
        plt.close()


def plot_attention_heatmap(attention_maps, gene_names, save_path):
    if not attention_maps:
        return
    stacked = torch.stack(attention_maps, dim=0)
    avg_map = tensor_to_numpy(stacked.mean(dim=(0, 1, 2)))
    if avg_map.shape[0] == len(gene_names) + 1:
        avg_map = avg_map[1:, 1:]
    plt.figure(figsize=(12, 10))
    sns.heatmap(avg_map, xticklabels=gene_names, yticklabels=gene_names, cmap='RdBu_r', center=avg_map.mean())
    plt.title('Transformer 注意力热图（候选基因之间的关系）')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def compute_gene_interaction_matrix(model, X, gene_names, device, save_csv_path=None, save_png_path=None):
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    n_genes = len(gene_names)
    interaction = np.zeros((n_genes, n_genes), dtype=float)

    with torch.no_grad():
        base_prob = tensor_to_numpy(torch.softmax(model(torch.FloatTensor(X).to(device)), dim=1)[:, 1])

        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                X_i = X.copy()
                X_j = X.copy()
                X_ij = X.copy()
                X_i[:, i] = 0
                X_j[:, j] = 0
                X_ij[:, i] = 0
                X_ij[:, j] = 0

                prob_i = tensor_to_numpy(torch.softmax(model(torch.FloatTensor(X_i).to(device)), dim=1)[:, 1])
                prob_j = tensor_to_numpy(torch.softmax(model(torch.FloatTensor(X_j).to(device)), dim=1)[:, 1])
                prob_ij = tensor_to_numpy(torch.softmax(model(torch.FloatTensor(X_ij).to(device)), dim=1)[:, 1])

                individual = np.mean(base_prob - prob_i) + np.mean(base_prob - prob_j)
                combined = np.mean(base_prob - prob_ij)
                interaction[i, j] = combined - individual
                interaction[j, i] = interaction[i, j]

    interaction_df = pd.DataFrame(interaction, index=gene_names, columns=gene_names)
    if save_csv_path:
        interaction_df.to_csv(save_csv_path)
    if save_png_path:
        plt.figure(figsize=(12, 10))
        sns.heatmap(interaction_df, cmap='coolwarm', center=0)
        plt.title('Transformer 基因交互矩阵（非线性关系示意）')
        plt.tight_layout()
        plt.savefig(save_png_path, dpi=300)
        plt.close()
    return interaction_df
