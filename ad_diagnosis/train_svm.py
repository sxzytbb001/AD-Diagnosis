import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .common import load_training_data, save_json

config.ensure_dirs()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def build_svm_model(params):
    return SVC(
        kernel=params.get('kernel', 'rbf'),
        C=params.get('C', 1.0),
        gamma=params.get('gamma', 'scale'),
        degree=params.get('degree', 3),
        probability=True,
        class_weight='balanced',
        random_state=42
    )


def select_best_svm_via_grid(X, y):
    print('\n>>> 步骤2: 5折 GridSearchCV 搜索最佳 SVM 参数...')
    base_model = SVC(probability=True, class_weight='balanced', random_state=42)
    param_grid = {
        'C': [0.1, 0.5, 1, 2, 5, 10, 20],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 0.5],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv5,
        n_jobs=-1,
        verbose=0,
        return_train_score=False
    )
    grid.fit(X, y)

    ranking_df = pd.DataFrame(grid.cv_results_)
    ranking_df = ranking_df.sort_values('mean_test_score', ascending=False)
    ranking_df.to_csv(os.path.join(config.SVM_DIR, 'grid_search_results.csv'), index=False)

    best_params = grid.best_params_
    best_score = float(grid.best_score_)
    print(f'  最佳参数: {best_params}')
    print(f'  5折均值AUC: {best_score:.4f}')
    return best_params, best_score, ranking_df


def build_voting_ensemble(best_params):
    C = best_params.get('C', 1.0)
    gamma = best_params.get('gamma', 'scale')

    voting = VotingClassifier(
        estimators=[
            ('svm_rbf', SVC(kernel='rbf', C=C, gamma=gamma, probability=True, class_weight='balanced', random_state=42)),
            ('svm_poly', SVC(kernel='poly', C=max(0.5, C), gamma=gamma, degree=3, probability=True, class_weight='balanced', random_state=42)),
            ('svm_linear', SVC(kernel='linear', C=C, probability=True, class_weight='balanced', random_state=42)),
            ('svm_sigmoid', SVC(kernel='sigmoid', C=C, gamma=gamma, probability=True, class_weight='balanced', random_state=42))
        ],
        voting='soft',
        weights=[3, 2, 1, 1],
        n_jobs=-1
    )
    return voting


def build_bagging_svm(best_params):
    base = SVC(
        kernel=best_params.get('kernel', 'rbf'),
        C=best_params.get('C', 1.0),
        gamma=best_params.get('gamma', 'scale'),
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    return BaggingClassifier(
        estimator=base,
        n_estimators=10,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )


def _calc_best_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[best_idx]), fpr, tpr


def cross_validate_models(X, y, best_params):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = {
        'SVM': np.zeros(len(y), dtype=float),
        'Voting_SVM': np.zeros(len(y), dtype=float),
        'Bagging_SVM': np.zeros(len(y), dtype=float)
    }

    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]

        model_single = build_svm_model(best_params)
        model_voting = build_voting_ensemble(best_params)
        model_bagging = build_bagging_svm(best_params)

        model_single.fit(X_train, y_train)
        model_voting.fit(X_train, y_train)
        model_bagging.fit(X_train, y_train)

        prob_single = model_single.predict_proba(X_val)[:, 1]
        prob_voting = model_voting.predict_proba(X_val)[:, 1]
        prob_bagging = model_bagging.predict_proba(X_val)[:, 1]

        oof_probs['SVM'][val_idx] = prob_single
        oof_probs['Voting_SVM'][val_idx] = prob_voting
        oof_probs['Bagging_SVM'][val_idx] = prob_bagging

        auc_single = roc_auc_score(y_val, prob_single)
        auc_voting = roc_auc_score(y_val, prob_voting)
        auc_bagging = roc_auc_score(y_val, prob_bagging)

        print(f'  Fold {fold}: SVM={auc_single:.4f}, Voting={auc_voting:.4f}, Bagging={auc_bagging:.4f}')

        fpr_v, tpr_v, _ = roc_curve(y_val, prob_voting)
        plt.plot(fpr_v, tpr_v, lw=1.0, alpha=0.35, label=f'Fold {fold} Voting (AUC={auc_voting:.3f})')

    summary = {}
    for name, probs in oof_probs.items():
        threshold, fpr, tpr = _calc_best_threshold(y, probs)
        auc_score = auc(fpr, tpr)
        pred = (probs >= threshold).astype(int)
        acc_score = accuracy_score(y, pred)
        summary[name] = {
            'oof_prob': probs,
            'auc': float(auc_score),
            'acc': float(acc_score),
            'threshold': float(threshold),
            'fpr': fpr,
            'tpr': tpr
        }

    plt.plot(summary['SVM']['fpr'], summary['SVM']['tpr'], lw=2, label=f'SVM OOF (AUC={summary["SVM"]["auc"]:.3f})')
    plt.plot(summary['Voting_SVM']['fpr'], summary['Voting_SVM']['tpr'], lw=2, label=f'Voting OOF (AUC={summary["Voting_SVM"]["auc"]:.3f})')
    plt.plot(summary['Bagging_SVM']['fpr'], summary['Bagging_SVM']['tpr'], lw=2, label=f'Bagging OOF (AUC={summary["Bagging_SVM"]["auc"]:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM 5折交叉验证 ROC')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.SVM_DIR, 'svm_roc.png'), dpi=300)
    plt.close()

    return summary


def train_svm():
    print('=' * 60)
    print('SVM 训练：5折网格搜索 + Voting + Bagging')
    print('=' * 60)

    candidate_genes = config.load_candidate_genes()
    if not candidate_genes:
        print('错误：未找到 候选基因，请先运行 feature_selection.py')
        return

    X, y, available_genes, _, _ = load_training_data(candidate_genes)
    X = np.asarray(X, dtype=np.float32)
    print(f'加载候选基因: {len(available_genes)} 个')
    print(f'训练数据: {X.shape[0]} 样本, {X.shape[1]} 基因')
    print(f'类别分布: AD={y.sum()}, Control={len(y) - y.sum()}')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_params, best_grid_auc, ranking_df = select_best_svm_via_grid(X_scaled, y)
    cv_summary = cross_validate_models(X, y, best_params)

    print('\n>>> 步骤3: 训练最终模型并保存...')
    final_single = build_svm_model(best_params)
    final_voting = build_voting_ensemble(best_params)
    final_bagging = build_bagging_svm(best_params)

    final_single.fit(X_scaled, y)
    final_voting.fit(X_scaled, y)
    final_bagging.fit(X_scaled, y)

    with open(config.SVM_MODEL_PATH, 'wb') as f:
        pickle.dump(final_single, f)
    with open(config.SVM_VOTING_PATH, 'wb') as f:
        pickle.dump(final_voting, f)
    with open(config.SVM_BAGGING_PATH, 'wb') as f:
        pickle.dump(final_bagging, f)
    with open(config.SVM_ENSEMBLE_PATH, 'wb') as f:
        pickle.dump({'voting': final_voting, 'bagging': final_bagging}, f)
    with open(config.SVM_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    params_to_save = {
        'best_grid_params': best_params,
        'best_grid_auc': best_grid_auc,
        'search_space': {
            'C': [0.1, 0.5, 1, 2, 5, 10, 20],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 0.5],
            'kernel': ['rbf', 'poly', 'sigmoid']
        },
        'thresholds': {
            'SVM': cv_summary['SVM']['threshold'],
            'Voting_SVM': cv_summary['Voting_SVM']['threshold'],
            'Bagging_SVM': cv_summary['Bagging_SVM']['threshold']
        },
        'oof_metrics': {
            'SVM': {'auc': cv_summary['SVM']['auc'], 'acc': cv_summary['SVM']['acc']},
            'Voting_SVM': {'auc': cv_summary['Voting_SVM']['auc'], 'acc': cv_summary['Voting_SVM']['acc']},
            'Bagging_SVM': {'auc': cv_summary['Bagging_SVM']['auc'], 'acc': cv_summary['Bagging_SVM']['acc']}
        },
        'candidate_gene_count': len(available_genes)
    }
    save_json(params_to_save, config.SVM_PARAMS_PATH)

    comparison_df = pd.DataFrame([
        {'Model': 'SVM', 'AUC': cv_summary['SVM']['auc'], 'Accuracy': cv_summary['SVM']['acc']},
        {'Model': 'Voting_SVM', 'AUC': cv_summary['Voting_SVM']['auc'], 'Accuracy': cv_summary['Voting_SVM']['acc']},
        {'Model': 'Bagging_SVM', 'AUC': cv_summary['Bagging_SVM']['auc'], 'Accuracy': cv_summary['Bagging_SVM']['acc']}
    ])
    comparison_df.to_csv(os.path.join(config.SVM_DIR, 'model_comparison.csv'), index=False)

    print('\n>>> 步骤4: 生成可视化结果...')
    single_prob = final_single.predict_proba(X_scaled)[:, 1]
    single_pred = (single_prob >= cv_summary['SVM']['threshold']).astype(int)
    cm = confusion_matrix(y, single_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Control', 'AD'], yticklabels=['Control', 'AD'])
    plt.title('SVM 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(os.path.join(config.SVM_DIR, 'svm_confusion_matrix.png'), dpi=300)
    plt.close()

    perm = permutation_importance(final_single, X_scaled, y, n_repeats=20, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({
        'gene': available_genes,
        'importance': perm.importances_mean,
        'std': perm.importances_std
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(config.SVM_DIR, 'svm_feature_importance.csv'), index=False)

    top_df = importance_df.head(15)
    plt.figure(figsize=(10, 7))
    plt.barh(range(len(top_df)), top_df['importance'].values, xerr=top_df['std'].values, color='#4c72b0', alpha=0.8)
    plt.yticks(range(len(top_df)), top_df['gene'].values)
    plt.gca().invert_yaxis()
    plt.xlabel('Permutation Importance')
    plt.title('SVM 特征重要性 Top15')
    plt.tight_layout()
    plt.savefig(os.path.join(config.SVM_DIR, 'svm_feature_importance.png'), dpi=300)
    plt.close()

    print('\n>>> 训练结果汇总')
    print(f"GridSearch 最佳参数: {best_params}, AUC={best_grid_auc:.4f}")
    print(f"SVM OOF AUC={cv_summary['SVM']['auc']:.4f}, Acc={cv_summary['SVM']['acc']:.4f}")
    print(f"Voting OOF AUC={cv_summary['Voting_SVM']['auc']:.4f}, Acc={cv_summary['Voting_SVM']['acc']:.4f}")
    print(f"Bagging OOF AUC={cv_summary['Bagging_SVM']['auc']:.4f}, Acc={cv_summary['Bagging_SVM']['acc']:.4f}")
    print(f'结果保存至: {config.SVM_DIR}')


if __name__ == '__main__':
    train_svm()
