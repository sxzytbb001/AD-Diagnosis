import copy
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .common import (
    EarlyStopping,
    FocalLoss,
    TransformerV3,
    compute_gene_interaction_matrix,
    load_training_data,
    plot_attention_heatmap,
    plot_tsne,
    save_json,
    tensor_to_numpy,
)

config.ensure_dirs()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def build_transformer(input_dim, params):
    return TransformerV3(
        input_dim=input_dim,
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        num_classes=2
    ).to(config.DEVICE)


def build_class_weights(y):
    counts = np.bincount(y.astype(int), minlength=2)
    weights = np.sqrt(len(y) / (2.0 * np.maximum(counts, 1)))
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=config.DEVICE)


def _augment_batch(batch_X, noise_std=0.15, mask_prob=0.10, epoch=0, warmup_epochs=10):
    """训练时数据增强：高斯噪声 + 随机特征遮蔽 + 特征缩放扰动。

    增加 warmup 机制：前 warmup_epochs 个 epoch 逐步增加增强强度，
    让模型先学到干净数据的主要模式，再通过增强提升鲁棒性。
    """
    # warmup：线性增加增强强度
    if epoch < warmup_epochs:
        ratio = epoch / max(warmup_epochs, 1)
    else:
        ratio = 1.0

    # 高斯噪声
    actual_noise = noise_std * ratio
    if actual_noise > 0:
        noise = torch.randn_like(batch_X) * actual_noise
        batch_X = batch_X + noise

    # 随机特征遮蔽
    actual_mask = mask_prob * ratio
    if actual_mask > 0:
        mask = (torch.rand_like(batch_X) > actual_mask).float()
        batch_X = batch_X * mask

    # 特征级缩放扰动
    scale_range = 0.10 * ratio
    if scale_range > 0:
        scale = 1.0 + (torch.rand(1, batch_X.shape[1], device=batch_X.device) - 0.5) * scale_range
        batch_X = batch_X * scale

    return batch_X


def train_single_model(model, X_train, y_train, X_val, y_val, params,
                       use_augmentation=True, label_smoothing=0.05):
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=False
    )

    class_weights = build_class_weights(y_train)
    ce_smooth = torch.nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smoothing
    )
    focal_aux = FocalLoss(alpha=class_weights, gamma=1.5)
    optimizer = AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    patience = params.get('patience', 20)
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4, mode='max')

    X_val_tensor = torch.FloatTensor(X_val).to(config.DEVICE)
    max_epochs = params['max_epochs']
    warmup_epochs = min(15, max_epochs // 8)

    # SWA: 收集最后阶段的模型权重进行平均
    swa_states = []
    swa_start_epoch = max(max_epochs // 2, 30)

    for epoch in range(max_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)

            if use_augmentation:
                batch_X = _augment_batch(
                    batch_X, noise_std=0.08, mask_prob=0.05,
                    epoch=epoch, warmup_epochs=warmup_epochs
                )

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = 0.85 * ce_smooth(logits, batch_y) + 0.15 * focal_aux(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step(epoch + 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_prob = tensor_to_numpy(torch.softmax(val_logits, dim=1)[:, 1])
            val_auc = roc_auc_score(y_val, val_prob)

        # SWA: 收集后半段的权重
        if epoch >= swa_start_epoch and epoch % 3 == 0:
            swa_states.append(copy.deepcopy(model.state_dict()))

        if early_stopping.step(val_auc, model):
            break

    # 如果有足够的 SWA 快照，用 SWA 平均；否则用 early stopping 的最佳
    if len(swa_states) >= 3:
        avg_state = {}
        for key in swa_states[0]:
            avg_state[key] = torch.stack([s[key].float() for s in swa_states]).mean(dim=0)
        model.load_state_dict(avg_state)
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            swa_prob = tensor_to_numpy(torch.softmax(val_logits, dim=1)[:, 1])
            swa_auc = roc_auc_score(y_val, swa_prob)
        # 只有 SWA 更好才使用
        if swa_auc >= early_stopping.best_score:
            early_stopping.best_score = swa_auc
            early_stopping.best_state = copy.deepcopy(model.state_dict())
        else:
            early_stopping.restore(model)
    else:
        early_stopping.restore(model)

    model.eval()
    with torch.no_grad():
        val_logits, val_features = model(X_val_tensor, return_features=True)
        val_prob = tensor_to_numpy(torch.softmax(val_logits, dim=1)[:, 1])
    return model, val_prob, tensor_to_numpy(val_features), early_stopping.best_score


def cross_validate_transformer(X, y, params, plot_prefix='transformer_cv'):
    """多种子交叉验证集成：使用多个随机种子运行 5 折 CV，
    将 OOF 概率取平均，有效降低模型方差。"""
    seeds = [42, 49, 84, 123, 256, 777, 2024]
    n_seeds = len(seeds)

    all_oof_probs = []
    all_oof_features = []
    all_fold_aucs = []

    for seed_idx, seed in enumerate(seeds):
        print(f'\n  === CV Seed {seed_idx + 1}/{n_seeds} (seed={seed}) ===')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_prob = np.zeros(len(y), dtype=float)
        oof_features = np.zeros((len(y), params['d_model']), dtype=float)
        fold_aucs = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
            X_train_raw, X_val_raw = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)

            torch.manual_seed(seed + fold)
            np.random.seed(seed + fold)

            model = build_transformer(X.shape[1], params)
            model, val_prob, val_features, val_auc = train_single_model(
                model, X_train, y_train, X_val, y_val, params
            )

            oof_prob[val_idx] = val_prob
            oof_features[val_idx] = val_features
            fold_aucs.append(val_auc)

            print(f'    Fold {fold}: AUC={val_auc:.4f}')

        seed_auc = roc_auc_score(y, oof_prob)
        print(f'  Seed {seed} OOF AUC: {seed_auc:.4f}')
        all_oof_probs.append(oof_prob)
        all_oof_features.append(oof_features)
        all_fold_aucs.extend(fold_aucs)

    # 多种子 OOF 平均
    ensemble_oof_prob = np.mean(all_oof_probs, axis=0)
    ensemble_oof_features = np.mean(all_oof_features, axis=0)

    fpr_mean, tpr_mean, thresholds = roc_curve(y, ensemble_oof_prob)
    auc_mean = auc(fpr_mean, tpr_mean)
    best_idx = np.argmax(tpr_mean - fpr_mean)
    best_threshold = thresholds[best_idx]

    precision, recall, pr_thresholds = precision_recall_curve(y, ensemble_oof_prob)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
    pr_best_idx = int(np.nanargmax(f1_scores))
    pr_best_threshold = pr_thresholds[pr_best_idx] if pr_best_idx < len(pr_thresholds) else best_threshold

    candidate_thresholds = np.unique(
        np.clip(
            np.concatenate([
                thresholds,
                pr_thresholds if len(pr_thresholds) else np.asarray([0.5]),
                np.asarray([0.5, best_threshold, pr_best_threshold]),
            ]),
            0.0,
            1.0,
        )
    )
    threshold_acc = []
    for thr in candidate_thresholds:
        pred_thr = (ensemble_oof_prob >= thr).astype(int)
        threshold_acc.append(accuracy_score(y, pred_thr))
    threshold_acc = np.asarray(threshold_acc)
    best_acc_idx = int(np.argmax(threshold_acc))
    acc_best_threshold = float(candidate_thresholds[best_acc_idx])

    oof_pred = (ensemble_oof_prob >= acc_best_threshold).astype(int)
    oof_acc = accuracy_score(y, oof_pred)

    print(f'\n  多种子集成 OOF AUC: {auc_mean:.4f}, Accuracy: {oof_acc:.4f}')

    # 绘制 ROC 曲线
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    for seed_idx, oof_p in enumerate(all_oof_probs):
        fpr_s, tpr_s, _ = roc_curve(y, oof_p)
        auc_s = auc(fpr_s, tpr_s)
        plt.plot(fpr_s, tpr_s, lw=1.0, alpha=0.4,
                 label=f'Seed {seeds[seed_idx]} OOF (AUC={auc_s:.3f})')
    plt.plot(fpr_mean, tpr_mean, 'b-', lw=2.5,
             label=f'Ensemble OOF (AUC={auc_mean:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Transformer 多种子集成 5折交叉验证 ROC')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.TRANSFORMER_DIR, f'{plot_prefix}_roc.png'), dpi=300)
    plt.close()

    return {
        'fold_aucs': all_fold_aucs,
        'fold_accs': [],
        'oof_prob': ensemble_oof_prob,
        'oof_pred': oof_pred,
        'oof_features': ensemble_oof_features,
        'auc_mean': auc_mean,
        'acc_mean': oof_acc,
        'oof_acc': oof_acc,
        'best_threshold': float(acc_best_threshold),
        'f1_threshold': float(pr_best_threshold),
        'roc_threshold': float(best_threshold)
    }


def _search_best_accuracy_threshold(y_true, y_prob, extra_thresholds=None):
    threshold_list = [np.unique(np.clip(y_prob, 0.0, 1.0))]
    if extra_thresholds is not None:
        threshold_list.append(np.unique(np.clip(np.asarray(extra_thresholds, dtype=float), 0.0, 1.0)))
    candidate_thresholds = np.unique(np.concatenate(threshold_list))
    best_acc = -1.0
    best_threshold = 0.5
    for threshold in candidate_thresholds:
        acc = accuracy_score(y_true, (y_prob >= threshold).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
    return best_threshold, float(best_acc)


def _train_final_ensemble_states(X, y, params, seeds):
    ensemble_states = []
    ensemble_configs = []
    for seed_i, seed in enumerate(seeds):
        params_i = params.copy()
        print(f'  训练集成模型 {seed_i + 1}/{len(seeds)} (seed={seed})')
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_idx, val_idx = train_test_split(
            np.arange(len(y)),
            test_size=0.15,
            stratify=y,
            random_state=seed
        )
        scaler_i = StandardScaler()
        X_train_i = scaler_i.fit_transform(X[train_idx])
        X_val_i = scaler_i.transform(X[val_idx])
        model_i = build_transformer(X.shape[1], params_i)
        model_i, _, _, _ = train_single_model(
            model_i, X_train_i, y[train_idx], X_val_i, y[val_idx], params_i
        )
        ensemble_states.append(copy.deepcopy(model_i.state_dict()))
        ensemble_configs.append(params_i)
    return ensemble_states, ensemble_configs


def _build_branch_artifact(X_branch, y, params, seeds):
    scaler = StandardScaler()
    scaler.fit(X_branch)
    states, configs = _train_final_ensemble_states(X_branch, y, params, seeds)
    return {
        'scaler': scaler,
        'states': states,
        'configs': configs,
    }


def train_transformer():
    print('=' * 60)
    print('Transformer 训练：轻量结构 + Focal Loss + 多种子集成')
    print('=' * 60)
    print(f'使用设备: {config.DEVICE}')

    candidate_genes = config.load_candidate_genes()
    if not candidate_genes:
        print('错误：未找到 候选基因，请先运行 feature_selection.py')
        return

    X, y, available_genes, _, _ = load_training_data(candidate_genes)
    X = np.asarray(X, dtype=np.float32)
    print(f'加载候选基因: {len(available_genes)} 个')
    print(f'训练数据: {X.shape[0]} 样本, {X.shape[1]} 基因')
    print(f'类别分布: AD={y.sum()}, Control={len(y) - y.sum()}')

    # 优化后的参数：降低 dropout，增加训练时间
    best_params = {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.18,
        'lr': 7e-4,
        'batch_size': 32,
        'weight_decay': 0.006,
        'max_epochs': 180,
        'patience': 24
    }
    print(f'\n>>> 步骤1: 使用优化参数: {best_params}')

    print('\n>>> 步骤2: 多种子交叉验证集成...')
    cv_result = cross_validate_transformer(X, y, best_params, plot_prefix='transformer')

    # 线性基线对比
    linear_cv_probs = np.zeros(len(y), dtype=float)
    linear_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in linear_cv.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        lr_model = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42)
        lr_model.fit(X_train, y[train_idx])
        linear_cv_probs[val_idx] = lr_model.predict_proba(X_val)[:, 1]

    linear_auc = roc_auc_score(y, linear_cv_probs)
    comparison_df = pd.DataFrame({
        'Model': ['Transformer', 'Logistic Regression'],
        'AUC': [cv_result['auc_mean'], linear_auc],
        'Accuracy': [cv_result['oof_acc'], accuracy_score(y, (linear_cv_probs >= 0.5).astype(int))]
    })
    comparison_df.to_csv(os.path.join(config.TRANSFORMER_DIR, 'model_comparison.csv'), index=False)

    print('\n>>> 步骤3: 训练多种子集成模型并保存...')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    seeds = [42, 49, 56, 63, 70, 77, 84, 91, 98, 123, 256, 333, 444, 777, 2024]
    ensemble_states, ensemble_configs = _train_final_ensemble_states(X, y, best_params, seeds)
    n_ensemble = len(seeds)

    print('\n>>> 步骤3.1: 训练外部泛化辅助分支(top20 / top28)...')
    adaptive_seeds = [42, 49, 56, 63, 70, 77, 84, 91, 123, 2024]
    adaptive_branch_20 = _build_branch_artifact(X[:, :20], y, best_params, adaptive_seeds)
    adaptive_branch_28 = _build_branch_artifact(X[:, :28], y, best_params, adaptive_seeds)

    # 保存主模型（第一个种子）
    model = build_transformer(X.shape[1], best_params)
    model.load_state_dict(ensemble_states[0])

    checkpoint = {
        'model_state_dict': ensemble_states[0],
        'ensemble_states': ensemble_states,
        'ensemble_configs': [{
            'input_dim': X.shape[1],
            'd_model': p['d_model'],
            'nhead': p['nhead'],
            'num_layers': p['num_layers'],
            'dropout': p['dropout']
        } for p in ensemble_configs],
        'n_ensemble': n_ensemble,
        'model_config': {
            'input_dim': X.shape[1],
            'd_model': best_params['d_model'],
            'nhead': best_params['nhead'],
            'num_layers': best_params['num_layers'],
            'dropout': best_params['dropout']
        },
        'candidate_genes': available_genes,
        'external_adaptation': {
            'shift_threshold': 0.73,
            'high_shift_gene_count': 20,
            'low_shift_gene_count': 28,
            'high_shift_soft_blend': 0.45,
            'high_shift_soft_alpha': 0.50,
            'low_shift_confidence_beta': 6.0,
            'low_shift_consensus_blend': 0.20,
            'branches': {
                '20': {
                    'gene_count': 20,
                    'states': adaptive_branch_20['states'],
                    'configs': [{
                        'input_dim': 20,
                        'd_model': p['d_model'],
                        'nhead': p['nhead'],
                        'num_layers': p['num_layers'],
                        'dropout': p['dropout']
                    } for p in adaptive_branch_20['configs']]
                },
                '28': {
                    'gene_count': 28,
                    'states': adaptive_branch_28['states'],
                    'configs': [{
                        'input_dim': 28,
                        'd_model': p['d_model'],
                        'nhead': p['nhead'],
                        'num_layers': p['num_layers'],
                        'dropout': p['dropout']
                    } for p in adaptive_branch_28['configs']]
                }
            }
        },
        'best_threshold': cv_result['best_threshold'],
        'roc_threshold': cv_result['roc_threshold'],
        'cv_auc_mean': cv_result['auc_mean'],
        'cv_acc_mean': cv_result['oof_acc'],
        'cv_oof_prob': cv_result['oof_prob'],
        'cv_y_train': y.tolist()
    }
    torch.save(checkpoint, config.TRANSFORMER_MODEL_PATH)
    with open(config.TRANSFORMER_SCALER_PATH, 'wb') as f:
        pickle.dump({
            'main': scaler,
            'external_branches': {
                '20': adaptive_branch_20['scaler'],
                '28': adaptive_branch_28['scaler']
            }
        }, f)

    best_params_to_save = best_params.copy()
    best_params_to_save.update({
        'best_threshold': cv_result['best_threshold'],
        'roc_threshold': cv_result['roc_threshold'],
        'cv_auc_mean': cv_result['auc_mean'],
        'cv_acc_mean': cv_result['oof_acc'],
        'candidate_gene_count': len(available_genes),
        'external_adaptation': {
            'shift_threshold': 0.73,
            'high_shift_gene_count': 20,
            'low_shift_gene_count': 28,
            'high_shift_soft_blend': 0.45,
            'high_shift_soft_alpha': 0.50,
            'low_shift_confidence_beta': 6.0,
            'low_shift_consensus_blend': 0.20
        }
    })
    save_json(best_params_to_save, config.TRANSFORMER_PARAMS_PATH)

    print('\n>>> 步骤4: 生成训练集可解释性图表...')
    model.eval()
    with torch.no_grad():
        logits, features, attention_maps, gene_weights = model(
            torch.FloatTensor(X_scaled).to(config.DEVICE),
            return_attention=True,
            return_features=True,
            return_gene_weights=True
        )
        train_prob = tensor_to_numpy(torch.softmax(logits, dim=1)[:, 1])
        train_features = tensor_to_numpy(features)
        train_pred = (train_prob >= cv_result['best_threshold']).astype(int)

    cm = confusion_matrix(y, train_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Control', 'AD'], yticklabels=['Control', 'AD'])
    plt.title('Transformer 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(os.path.join(config.TRANSFORMER_DIR, 'transformer_confusion_matrix.png'), dpi=300)
    plt.close()

    plot_attention_heatmap(
        attention_maps,
        available_genes,
        os.path.join(config.TRANSFORMER_DIR, 'attention_heatmap.png')
    )
    gene_weight_mean = tensor_to_numpy(gene_weights.mean(dim=0))
    gate_importance_df = pd.DataFrame({
        'Gene': available_genes,
        'Weight': gene_weight_mean
    }).sort_values('Weight', ascending=False)
    gate_importance_df.to_csv(os.path.join(config.TRANSFORMER_DIR, 'transformer_gate_importance.csv'), index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=gate_importance_df.head(15), x='Weight', y='Gene', palette='viridis')
    plt.title('Transformer Gate Importance (Top 15 Genes)')
    plt.xlabel('Average gate weight')
    plt.ylabel('Gene')
    plt.tight_layout()
    plt.savefig(os.path.join(config.TRANSFORMER_DIR, 'transformer_gate_importance.png'), dpi=300)
    plt.close()
    compute_gene_interaction_matrix(
        model,
        X_scaled[: min(len(X_scaled), 80)],
        available_genes,
        config.DEVICE,
        save_csv_path=os.path.join(config.TRANSFORMER_DIR, 'gene_interaction_matrix.csv'),
        save_png_path=os.path.join(config.TRANSFORMER_DIR, 'gene_interaction_matrix.png')
    )
    plot_tsne(
        train_features,
        y,
        os.path.join(config.TRANSFORMER_DIR, 'transformer_tsne.png'),
        'Transformer 特征 t-SNE'
    )

    fpr, tpr, _ = roc_curve(y, cv_result['oof_prob'])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'Transformer OOF ROC (AUC={cv_result["auc_mean"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Transformer OOF ROC')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.TRANSFORMER_DIR, 'transformer_oof_roc.png'), dpi=300)
    plt.close()

    print('\n>>> 训练结果汇总')
    print(f'多种子集成 OOF AUC: {cv_result["auc_mean"]:.4f}')
    print(f'多种子集成 OOF Accuracy: {cv_result["oof_acc"]:.4f}')
    print(f'推荐阈值: {cv_result["best_threshold"]:.4f}')
    print(f'线性基线 Logistic AUC: {linear_auc:.4f}')
    print(f'模型与结果已保存至: {config.TRANSFORMER_DIR}')


if __name__ == '__main__':
    train_transformer()
