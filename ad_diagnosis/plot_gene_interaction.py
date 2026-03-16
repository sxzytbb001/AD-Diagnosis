"""Generate post-hoc Transformer gene interaction figures.

This script is intentionally kept outside the default pipeline because it is an
interpretability step, not a training requirement. It works with the current
TransformerV3 implementation and the saved ensemble checkpoint.
"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler

from . import config
from .common import TransformerV3, load_training_data, tensor_to_numpy, transform_with_preprocessor

warnings.filterwarnings("ignore")
config.ensure_dirs()

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = config.TRANSFORMER_DIR


def load_models_and_data():
    candidate_genes = config.load_candidate_genes()
    if not candidate_genes:
        raise FileNotFoundError("candidate_genes.txt not found. Run feature_selection.py first.")
    if not os.path.exists(config.TRANSFORMER_MODEL_PATH):
        raise FileNotFoundError("Transformer checkpoint not found. Run train_transformer.py first.")

    X, y, available_genes, _, _ = load_training_data(candidate_genes)
    X = np.asarray(X, dtype=np.float32)

    if os.path.exists(config.TRANSFORMER_SCALER_PATH):
        with open(config.TRANSFORMER_SCALER_PATH, "rb") as file_obj:
            scaler = pickle.load(file_obj)
            if isinstance(scaler, dict):
                scaler = scaler.get('main', scaler)
    else:
        scaler = StandardScaler().fit(X)
    X_scaled = transform_with_preprocessor(scaler, X)

    checkpoint = torch.load(config.TRANSFORMER_MODEL_PATH, map_location=config.DEVICE, weights_only=False)
    ensemble_states = checkpoint.get("ensemble_states") or [checkpoint["model_state_dict"]]
    ensemble_configs = checkpoint.get("ensemble_configs") or [checkpoint["model_config"]] * len(ensemble_states)

    models = []
    for state_dict, model_config in zip(ensemble_states, ensemble_configs):
        model = TransformerV3(
            input_dim=model_config["input_dim"],
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            num_classes=2,
        ).to(config.DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    return models, X_scaled, y, available_genes


def ensemble_predict(models, X):
    X_tensor = torch.FloatTensor(X).to(config.DEVICE)
    probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)
            probs.append(tensor_to_numpy(torch.softmax(logits, dim=1)[:, 1]))
    return np.mean(np.vstack(probs), axis=0)


def extract_attention_matrix(models, X_scaled, gene_count, max_samples=160):
    X_eval = X_scaled[: min(len(X_scaled), max_samples)]
    X_tensor = torch.FloatTensor(X_eval).to(config.DEVICE)
    attention_matrices = []

    for model in models:
        model.eval()
        with torch.no_grad():
            _, attention_maps = model(X_tensor, return_attention=True)

        if not attention_maps:
            continue

        per_layer = []
        for attn in attention_maps:
            matrix = tensor_to_numpy(attn.mean(dim=(0, 1)))
            if matrix.shape[0] == gene_count + 1:
                matrix = matrix[1:, 1:]
            per_layer.append(matrix)
        attention_matrices.append(np.mean(per_layer, axis=0))

    if not attention_matrices:
        return None
    return np.mean(attention_matrices, axis=0)


def compute_interaction_from_predictions(models, X_scaled, genes, max_samples=120):
    X_eval = np.asarray(X_scaled[: min(len(X_scaled), max_samples)], dtype=np.float32)
    n_genes = len(genes)
    interaction_matrix = np.zeros((n_genes, n_genes), dtype=float)
    marginal_effects = np.zeros(n_genes, dtype=float)

    base_prob = ensemble_predict(models, X_eval)

    for i in range(n_genes):
        X_i = X_eval.copy()
        X_i[:, i] = 0.0
        prob_i = ensemble_predict(models, X_i)
        marginal_effects[i] = float(np.mean(np.abs(base_prob - prob_i)))

    print(f"Computing pairwise interactions for {n_genes} genes on {len(X_eval)} samples...")
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            X_i = X_eval.copy()
            X_j = X_eval.copy()
            X_ij = X_eval.copy()

            X_i[:, i] = 0.0
            X_j[:, j] = 0.0
            X_ij[:, i] = 0.0
            X_ij[:, j] = 0.0

            prob_i = ensemble_predict(models, X_i)
            prob_j = ensemble_predict(models, X_j)
            prob_ij = ensemble_predict(models, X_ij)

            effect_i = np.mean(np.abs(base_prob - prob_i))
            effect_j = np.mean(np.abs(base_prob - prob_j))
            effect_ij = np.mean(np.abs(base_prob - prob_ij))
            interaction = abs(effect_ij - (effect_i + effect_j))

            interaction_matrix[i, j] = interaction
            interaction_matrix[j, i] = interaction

        if (i + 1) % 5 == 0 or i + 1 == n_genes:
            print(f"  progress: {i + 1}/{n_genes}")

    return interaction_matrix, marginal_effects


def build_pair_dataframe(interaction_matrix, X_scaled, genes):
    rows = []
    n_genes = len(genes)
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            corr = float(abs(np.corrcoef(X_scaled[:, i], X_scaled[:, j])[0, 1]))
            interaction = float(interaction_matrix[i, j])
            rows.append(
                {
                    "gene_a": genes[i],
                    "gene_b": genes[j],
                    "linear_corr_abs": corr,
                    "interaction": interaction,
                    "interaction_priority": interaction / (corr + 0.1),
                }
            )
    return pd.DataFrame(rows).sort_values("interaction", ascending=False)


def plot_attention_heatmap(attention_matrix, genes, save_path):
    if attention_matrix is None:
        return
    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_matrix, xticklabels=genes, yticklabels=genes, cmap="RdBu_r", center=attention_matrix.mean())
    plt.title("Transformer Ensemble Attention Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_interaction_network(interaction_df, save_path):
    top_pairs = interaction_df.head(30).reset_index(drop=True)
    if top_pairs.empty:
        return

    involved_genes = sorted(set(top_pairs["gene_a"]).union(top_pairs["gene_b"]))
    angles = np.linspace(0, 2 * np.pi, len(involved_genes), endpoint=False)
    positions = {gene: (np.cos(angle) * 4.2, np.sin(angle) * 4.2) for gene, angle in zip(involved_genes, angles)}

    fig, ax = plt.subplots(figsize=(12, 12))
    max_interaction = float(top_pairs["interaction"].max()) + 1e-8

    for _, row in top_pairs.iterrows():
        x1, y1 = positions[row["gene_a"]]
        x2, y2 = positions[row["gene_b"]]
        strength = row["interaction"] / max_interaction
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=plt.cm.Reds(0.35 + 0.6 * strength),
            alpha=0.35 + 0.55 * strength,
            linewidth=0.8 + 3.0 * strength,
            zorder=1,
        )

    for gene, (x_pos, y_pos) in positions.items():
        ax.scatter(x_pos, y_pos, s=850, c="#4C72B0", edgecolors="white", linewidth=2, zorder=2)
        ax.text(x_pos, y_pos, gene, fontsize=8, ha="center", va="center", color="white", fontweight="bold", zorder=3)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Top Transformer Gene Interaction Network", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_nonlinear_vs_linear(interaction_df, save_path):
    median_interaction = float(interaction_df["interaction"].median())
    plot_df = interaction_df.copy()
    plot_df["nonlinear_only"] = (plot_df["linear_corr_abs"] < 0.3) & (plot_df["interaction"] > median_interaction)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    scatter = axes[0].scatter(
        plot_df["linear_corr_abs"],
        plot_df["interaction"],
        c=plot_df["interaction"],
        cmap="YlOrRd",
        s=32,
        alpha=0.75,
        edgecolors="gray",
        linewidth=0.4,
    )
    plt.colorbar(scatter, ax=axes[0], label="Interaction strength")
    axes[0].axvline(0.3, color="gray", linestyle="--", alpha=0.5)
    axes[0].axhline(median_interaction, color="steelblue", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Absolute linear correlation")
    axes[0].set_ylabel("Transformer interaction strength")
    axes[0].set_title("Linear correlation vs nonlinear interaction")

    highlight_df = plot_df.sort_values("interaction_priority", ascending=False).head(12)
    for _, row in highlight_df.iterrows():
        if row["linear_corr_abs"] < 0.3:
            label = f"{row['gene_a']}-{row['gene_b']}"
            axes[0].annotate(label, (row["linear_corr_abs"], row["interaction"]), fontsize=6, color="darkred")

    category_counts = [
        int(((plot_df["linear_corr_abs"] < 0.3) & (plot_df["interaction"] > median_interaction)).sum()),
        int(((plot_df["linear_corr_abs"] >= 0.3) & (plot_df["interaction"] > median_interaction)).sum()),
        int(((plot_df["linear_corr_abs"] < 0.3) & (plot_df["interaction"] <= median_interaction)).sum()),
        int(((plot_df["linear_corr_abs"] >= 0.3) & (plot_df["interaction"] <= median_interaction)).sum()),
    ]
    category_labels = [
        "Low linear\nHigh interaction",
        "High linear\nHigh interaction",
        "Low linear\nLow interaction",
        "High linear\nLow interaction",
    ]
    category_colors = ["#e74c3c", "#f39c12", "#95a5a6", "#3498db"]
    bars = axes[1].bar(range(len(category_labels)), category_counts, color=category_colors, edgecolor="white", linewidth=1.5)
    axes[1].set_xticks(range(len(category_labels)))
    axes[1].set_xticklabels(category_labels, fontsize=9)
    axes[1].set_ylabel("Gene pair count")
    axes[1].set_title("Pair relationship categories")
    for bar, count in zip(bars, category_counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(count), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_gene_pairs(X_scaled, y, genes, interaction_df, save_path):
    top_pairs = interaction_df.sort_values("interaction_priority", ascending=False).head(6).reset_index(drop=True)
    gene_to_index = {gene: idx for idx, gene in enumerate(genes)}

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    ctrl_mask = y == 0
    ad_mask = y == 1

    for idx, row in top_pairs.iterrows():
        ax = axes[idx // 3][idx % 3]
        idx_a = gene_to_index[row["gene_a"]]
        idx_b = gene_to_index[row["gene_b"]]

        ax.scatter(
            X_scaled[ctrl_mask, idx_a],
            X_scaled[ctrl_mask, idx_b],
            c="#3498db",
            alpha=0.6,
            s=28,
            label="Control",
            edgecolors="white",
            linewidth=0.4,
        )
        ax.scatter(
            X_scaled[ad_mask, idx_a],
            X_scaled[ad_mask, idx_b],
            c="#e74c3c",
            alpha=0.6,
            s=28,
            label="AD",
            edgecolors="white",
            linewidth=0.4,
        )
        ax.set_xlabel(row["gene_a"])
        ax.set_ylabel(row["gene_b"])
        ax.set_title(f"interaction={row['interaction']:.4f}, |r|={row['linear_corr_abs']:.2f}", fontsize=9)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle("Top nonlinear gene pairs highlighted by Transformer", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(marginal_effects, genes, save_csv_path, save_png_path):
    importance_df = pd.DataFrame({"gene": genes, "importance": marginal_effects}).sort_values("importance", ascending=False)
    importance_df.to_csv(save_csv_path, index=False)

    plt.figure(figsize=(10, 10))
    sns.barplot(data=importance_df, x="importance", y="gene", palette="YlOrRd")
    plt.title("Transformer occlusion importance")
    plt.xlabel("Mean probability change after gene masking")
    plt.ylabel("Gene")
    plt.tight_layout()
    plt.savefig(save_png_path, dpi=300, bbox_inches="tight")
    plt.close()
    return importance_df


def main():
    print("=" * 60)
    print("Transformer gene interaction analysis")
    print("=" * 60)

    models, X_scaled, y, genes = load_models_and_data()
    print(f"Loaded {len(models)} Transformer model(s)")
    print(f"Samples: {X_scaled.shape[0]}, genes: {len(genes)}")

    attention_matrix = extract_attention_matrix(models, X_scaled, len(genes))
    if attention_matrix is not None:
        plot_attention_heatmap(attention_matrix, genes, os.path.join(OUTPUT_DIR, "ensemble_attention_heatmap.png"))

    interaction_matrix, marginal_effects = compute_interaction_from_predictions(models, X_scaled, genes)
    interaction_df = build_pair_dataframe(interaction_matrix, X_scaled, genes)

    matrix_df = pd.DataFrame(interaction_matrix, index=genes, columns=genes)
    matrix_df.to_csv(os.path.join(OUTPUT_DIR, "nonlinear_interaction_matrix.csv"))
    interaction_df.to_csv(os.path.join(OUTPUT_DIR, "nonlinear_interaction_pairs.csv"), index=False)

    plot_interaction_network(interaction_df, os.path.join(OUTPUT_DIR, "gene_interaction_network.png"))
    plot_nonlinear_vs_linear(interaction_df, os.path.join(OUTPUT_DIR, "nonlinear_vs_linear_comparison.png"))
    plot_top_gene_pairs(X_scaled, y, genes, interaction_df, os.path.join(OUTPUT_DIR, "top_nonlinear_gene_pairs.png"))
    importance_df = plot_feature_importance(
        marginal_effects,
        genes,
        os.path.join(OUTPUT_DIR, "transformer_feature_importance.csv"),
        os.path.join(OUTPUT_DIR, "transformer_feature_importance.png"),
    )

    print("\nTop 5 interaction pairs:")
    print(interaction_df.head(5)[["gene_a", "gene_b", "interaction", "linear_corr_abs"]].to_string(index=False))
    print("\nTop 5 occlusion-important genes:")
    print(importance_df.head(5).to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
