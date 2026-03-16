import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import config
from .common import benjamini_hochberg, cohens_d, load_training_matrix_and_labels

config.ensure_dirs()
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def _rank_score(rank_series, top_k, weight):
    scores = {}
    for rank, gene in enumerate(rank_series[:top_k], start=1):
        scores[gene] = weight * (top_k - rank + 1) / top_k
    return scores


def _evaluate_gene_set_auc(X_df, y, genes, n_splits=3, max_genes=30):
    genes = [g for g in genes if g in X_df.columns][:max_genes]
    if len(genes) < 3:
        return 0.50

    X = X_df[genes].values
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]

        clf = LogisticRegression(
            max_iter=3000,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_prob))

    return float(np.mean(aucs))


def _mrmr_ranking(X_df, y, top_k=100, max_candidates=300):
    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(X_df, y, discrete_features=False, random_state=42)
    mi_series = pd.Series(mi, index=X_df.columns).sort_values(ascending=False)
    candidate_genes = mi_series.head(max_candidates).index.tolist()

    corr_matrix = X_df[candidate_genes].corr().abs().fillna(0.0)
    selected = []
    scores = {}

    for _ in range(min(top_k, len(candidate_genes))):
        best_gene = None
        best_score = -np.inf
        for gene in candidate_genes:
            if gene in selected:
                continue
            relevance = mi_series.loc[gene]
            redundancy = corr_matrix.loc[gene, selected].mean() if selected else 0.0
            score = relevance - redundancy
            if score > best_score:
                best_score = score
                best_gene = gene
        if best_gene is None:
            break
        selected.append(best_gene)
        scores[best_gene] = best_score

    ranking = pd.Series(scores).sort_values(ascending=False)
    return ranking.index.tolist(), ranking


def _stability_selection(
    X_df,
    y,
    top_k=100,
    n_iterations=50,
    threshold=0.8,
    sample_ratio=0.8,
    max_iter=1500,
    tol=1e-3,
    verbose_every=5,
):
    rng = np.random.default_rng(42)
    counts = pd.Series(0.0, index=X_df.columns)

    print(f'  Stability Selection 启动: 特征={X_df.shape[1]}, 迭代={n_iterations}, 采样比例={sample_ratio}')
    t0 = time.time()

    for i in range(n_iterations):
        idx = rng.choice(len(y), size=int(len(y) * sample_ratio), replace=True)
        X_sub = X_df.iloc[idx]
        y_sub = y[idx]

        model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5,
            C=1.0,
            max_iter=max_iter,
            tol=tol,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_sub, y_sub)
        non_zero = (np.abs(model.coef_).ravel() > 1e-8).astype(float)
        counts += pd.Series(non_zero, index=X_df.columns)

        if (i + 1) % max(1, verbose_every) == 0 or i == 0 or (i + 1) == n_iterations:
            elapsed = time.time() - t0
            print(f'    Stability 进度: {i + 1}/{n_iterations} (累计 {elapsed:.1f}s)')

    freq = counts / float(n_iterations)
    selected = freq[freq >= threshold].sort_values(ascending=False)

    if len(selected) < top_k:
        ranking = freq.sort_values(ascending=False).head(top_k)
    else:
        ranking = selected.head(top_k)

    return ranking.index.tolist(), freq


def _xgboost_gain_ranking(X_df, y, top_k=100):
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_df.values, y)

        booster = model.get_booster()
        gain_map = booster.get_score(importance_type='gain')

        score_series = pd.Series(0.0, index=X_df.columns)
        for i, gene in enumerate(X_df.columns):
            score_series.loc[gene] = float(gain_map.get(f'f{i}', 0.0))

        score_series = score_series.sort_values(ascending=False)
        return score_series.head(top_k).index.tolist(), score_series, 'XGBoost'

    except Exception:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_df, y)

        score_series = pd.Series(model.feature_importances_, index=X_df.columns).sort_values(ascending=False)
        return score_series.head(top_k).index.tolist(), score_series, 'XGBoost(Fallback-RF)'


def feature_selection():
    print('=' * 60)
    print('集成特征选择：七方法+性能加权投票')
    print('=' * 60)

    gene_matrix, _, y = load_training_matrix_and_labels()
    X = gene_matrix.copy()
    print(f'训练数据: {X.shape[0]} 样本, {X.shape[1]} 基因')

    group_ad = X.iloc[y == 1]
    group_ctrl = X.iloc[y == 0]

    print('\n>>> 步骤1: 差异分析初筛（保留 Top-1200）...')
    t_stats, p_vals = stats.ttest_ind(group_ad, group_ctrl, axis=0, equal_var=False, nan_policy='omit')
    mean_ad = group_ad.mean(axis=0)
    mean_ctrl = group_ctrl.mean(axis=0)
    effect_sizes = [cohens_d(group_ad[col].values, group_ctrl[col].values) for col in X.columns]
    fdr_vals = benjamini_hochberg(np.nan_to_num(p_vals, nan=1.0))

    stats_df = pd.DataFrame({
        'gene': X.columns,
        't_stat': np.nan_to_num(t_stats, nan=0.0),
        'p_value': np.nan_to_num(p_vals, nan=1.0),
        'fdr': fdr_vals,
        'cohens_d': effect_sizes,
        'mean_ad': mean_ad.values,
        'mean_control': mean_ctrl.values,
        'abs_t_stat': np.abs(np.nan_to_num(t_stats, nan=0.0)),
        'abs_effect_size': np.abs(effect_sizes)
    }).sort_values(['p_value', 'abs_effect_size'], ascending=[True, False])

    prefilter_genes = stats_df.head(1200)['gene'].tolist()
    X_prefilter = X[prefilter_genes]
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_prefilter), columns=prefilter_genes, index=X.index)
    print(f'初筛后保留基因数: {len(prefilter_genes)}')

    print('\n>>> 步骤2: 七方法 Top-100 排序...')
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier

    mi_scores = mutual_info_classif(X_scaled, y, discrete_features=False, random_state=42)
    mi_df = pd.DataFrame({'gene': prefilter_genes, 'score': mi_scores}).sort_values('score', ascending=False)
    mi_top = mi_df.head(100)['gene'].tolist()
    print('  Mutual Information Top100 完成')

    # 稳定性选择在 MI 预筛后的子空间中进行，避免在 1200 维上迭代 50 次导致耗时过长
    stability_candidate_genes = mi_df.head(400)['gene'].tolist()

    xgb_top, xgb_series, xgb_method_name = _xgboost_gain_ranking(X_scaled, y, top_k=100)
    print(f'  {xgb_method_name} Top100 完成')

    enet = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty='elasticnet',
        solver='saga',
        l1_ratios=[0.5],
        scoring='roc_auc',
        max_iter=5000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    enet.fit(X_scaled, y)
    enet_series = pd.Series(np.abs(enet.coef_).ravel(), index=prefilter_genes).sort_values(ascending=False)
    enet_top = enet_series.head(100).index.tolist()
    print('  Elastic Net Top100 完成')

    mrmr_top, mrmr_series = _mrmr_ranking(X_scaled, y, top_k=100, max_candidates=300)
    print('  mRMR Top100 完成')

    stability_top, stability_freq = _stability_selection(
        X_scaled[stability_candidate_genes],
        y,
        top_k=100,
        n_iterations=50,
        threshold=0.8,
        sample_ratio=0.8,
        max_iter=1500,
        tol=1e-3,
        verbose_every=5,
    )
    print('  Stability Selection Top100 完成 (50次, 阈值0.8)')

    # --- 新增方法6: T-test 排序 ---
    ttest_ranking = stats_df[stats_df['gene'].isin(prefilter_genes)].sort_values(
        'abs_t_stat', ascending=False
    ).head(100)['gene'].tolist()
    print('  T-test Top100 完成')

    # --- 新增方法7: Random Forest 重要性 ---
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_scaled, y)
    rf_series = pd.Series(rf_model.feature_importances_, index=prefilter_genes).sort_values(ascending=False)
    rf_top = rf_series.head(100).index.tolist()
    print('  Random Forest Top100 完成')

    method_rankings = {
        'TTest': ttest_ranking,
        'MutualInfo': mi_top,
        'XGBoost': xgb_top,
        'RandomForest': rf_top,
        'ElasticNet': enet_top,
        'mRMR': mrmr_top,
        'Stability': stability_top
    }

    print('\n>>> 步骤3: 基于验证性能计算方法权重...')
    method_perf = {}
    for method, ranking in method_rankings.items():
        auc_score = _evaluate_gene_set_auc(X_scaled, y, ranking, n_splits=3, max_genes=30)
        method_perf[method] = auc_score
        print(f'  {method:<10} 验证AUC={auc_score:.4f}')

    perf_series = pd.Series(method_perf)
    raw_weights = np.clip(perf_series.values - 0.5, 1e-3, None)
    norm_weights = raw_weights / np.sum(raw_weights)
    method_weights = dict(zip(perf_series.index.tolist(), norm_weights.tolist()))

    print('  方法权重:')
    for k, v in method_weights.items():
        print(f'    - {k:<10}: {v:.3f}')

    print('\n>>> 步骤4: 加权投票融合...')
    score_map = {gene: 0.0 for gene in prefilter_genes}
    for method, ranking in method_rankings.items():
        scores = _rank_score(ranking, top_k=100, weight=method_weights[method])
        for gene, s in scores.items():
            score_map[gene] += s

    vote_count = {gene: sum([gene in method_rankings[m] for m in method_rankings]) for gene in prefilter_genes}

    merged_df = stats_df.copy()
    merged_df['mi_score'] = merged_df['gene'].map(mi_df.set_index('gene')['score']).fillna(0)
    merged_df['xgb_gain'] = merged_df['gene'].map(xgb_series).fillna(0)
    merged_df['elastic_net_coef'] = merged_df['gene'].map(enet_series).fillna(0)
    merged_df['mrmr_score'] = merged_df['gene'].map(mrmr_series).fillna(0)
    merged_df['stability_freq'] = merged_df['gene'].map(stability_freq).fillna(0)
    merged_df['vote_score'] = merged_df['gene'].map(score_map).fillna(0)
    merged_df['vote_count'] = merged_df['gene'].map(vote_count).fillna(0).astype(int)

    merged_df['quality_score'] = (
        0.45 * merged_df['vote_score'] +
        0.20 * (merged_df['abs_t_stat'] / (merged_df['abs_t_stat'].max() + 1e-8)) +
        0.20 * (merged_df['abs_effect_size'] / (merged_df['abs_effect_size'].max() + 1e-8)) +
        0.15 * (1 - merged_df['fdr'])
    )

    filtered_df = merged_df[(merged_df['vote_count'] >= 2) & (merged_df['fdr'] < 0.2)].copy()
    if len(filtered_df) < 20:
        filtered_df = merged_df[merged_df['vote_count'] >= 2].copy()
    if len(filtered_df) < 20:
        filtered_df = merged_df.copy()

    final_df = filtered_df.sort_values(['quality_score', 'vote_count', 'abs_effect_size'], ascending=False).head(30).copy()
    candidate_genes = final_df['gene'].tolist()

    print(f'最终候选基因 ({len(candidate_genes)}个):')
    print(candidate_genes)

    print('\n>>> 步骤5: 保存结果...')
    with open(config.CANDIDATE_GENES_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(candidate_genes))

    final_df.to_csv(config.FEATURE_STATS_PATH, index=False)
    merged_df.sort_values('quality_score', ascending=False).to_csv(
        os.path.join(config.FEATURE_SELECTION_DIR, 'feature_selection_ranking.csv'),
        index=False
    )

    method_df = pd.DataFrame({
        'method': list(method_rankings.keys()),
        'validation_auc': [method_perf[m] for m in method_rankings.keys()],
        'weight': [method_weights[m] for m in method_rankings.keys()]
    }).sort_values('validation_auc', ascending=False)
    method_df.to_csv(os.path.join(config.FEATURE_SELECTION_DIR, 'method_weights.csv'), index=False)

    method_top_files = {
        'top_ttest.txt': ttest_ranking,
        'top_mutual_info.txt': mi_top,
        'top_xgboost.txt': xgb_top,
        'top_random_forest.txt': rf_top,
        'top_elastic_net.txt': enet_top,
        'top_mrmr.txt': mrmr_top,
        'top_stability.txt': stability_top
    }
    for file_name, genes in method_top_files.items():
        with open(os.path.join(config.FEATURE_SELECTION_DIR, file_name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(genes))

    print('\n>>> 步骤6: 绘图展示...')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=method_df, x='method', y='validation_auc', palette='Set2')
    plt.title('特征选择方法验证AUC')
    plt.ylim(max(0.45, method_df['validation_auc'].min() - 0.05), min(1.0, method_df['validation_auc'].max() + 0.05))
    plt.tight_layout()
    plt.savefig(os.path.join(config.FEATURE_SELECTION_DIR, 'method_overview.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.barplot(data=final_df, x='quality_score', y='gene', palette='viridis')
    plt.title('最终候选基因综合得分')
    plt.xlabel('综合得分')
    plt.ylabel('基因')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FEATURE_SELECTION_DIR, 'candidate_gene_scores.png'), dpi=300)
    plt.close()

    heatmap_data = X[candidate_genes].T
    plt.figure(figsize=(12, 9))
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=heatmap_data.values.mean())
    plt.title('候选基因表达热图')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FEATURE_SELECTION_DIR, 'candidate_gene_heatmap.png'), dpi=300)
    plt.close()

    print('\n' + '=' * 60)
    print('特征选择完成，结果保存至:')
    print(config.FEATURE_SELECTION_DIR)
    print('=' * 60)


if __name__ == '__main__':
    feature_selection()
