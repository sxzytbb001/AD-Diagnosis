"""Generate candidate-gene expression boxplots across datasets."""

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from . import config

warnings.filterwarnings("ignore")
config.ensure_dirs()

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = config.GENE_VALIDATION_DIR

DATASETS = {
    "Train_GSE33000": {"dir": config.TRAIN_DATA_DIR, "type": "train"},
    "Val_GSE122063": {"dir": config.EXTERNAL_DATA_DIRS["GSE122063"], "type": "external"},
    "Val_GSE109887": {"dir": config.EXTERNAL_DATA_DIRS["GSE109887"], "type": "external"},
}


def load_dataset(name, ds_config):
    print(f"Loading {name}...")
    data_dir = ds_config["dir"]

    if ds_config["type"] == "train":
        matrix_path = os.path.join(data_dir, "cleaned_gene_matrix.csv")
        labels_path = os.path.join(data_dir, "sample_labels.csv")
        if not os.path.exists(matrix_path) or not os.path.exists(labels_path):
            return None

        df = pd.read_csv(matrix_path, index_col=0).T
        labels = pd.read_csv(labels_path)
        df = df.loc[labels["sample_id"]]
        df["Group"] = labels["label"].values
        return df

    matrix_path = os.path.join(data_dir, "geneMatrix.txt")
    s1_path = os.path.join(data_dir, "s1.txt")
    s2_path = os.path.join(data_dir, "s2.txt")
    if not os.path.exists(matrix_path) or not os.path.exists(s1_path) or not os.path.exists(s2_path):
        return None

    try:
        full_df = pd.read_csv(matrix_path, sep="\t", index_col=0)
    except Exception:
        full_df = pd.read_csv(matrix_path, sep=r"\s+", index_col=0)

    with open(s1_path, "r", encoding="utf-8") as file_obj:
        s1 = [line.strip() for line in file_obj if line.strip()]
    with open(s2_path, "r", encoding="utf-8") as file_obj:
        s2 = [line.strip() for line in file_obj if line.strip()]

    valid_s1 = [sample for sample in s1 if sample in full_df.columns]
    valid_s2 = [sample for sample in s2 if sample in full_df.columns]

    df_s1 = full_df[valid_s1].T
    df_s1["Group"] = "Control"
    df_s2 = full_df[valid_s2].T
    df_s2["Group"] = "AD"
    return pd.concat([df_s1, df_s2], axis=0)


def plot_gene_boxplots(all_data, candidate_genes):
    print("\n>>> Generating gene expression boxplots...")
    sns.set_style("whitegrid")

    for gene in candidate_genes:
        print(f"  processing gene: {gene}")
        plot_frames = []

        for dataset_name, df in all_data.items():
            if gene not in df.columns:
                print(f"    missing gene in {dataset_name}: {gene}")
                continue

            temp_df = df[[gene, "Group"]].copy()
            temp_df["Dataset"] = dataset_name
            plot_frames.append(temp_df)

        if not plot_frames:
            continue

        final_df = pd.concat(plot_frames, axis=0)
        plt.figure(figsize=(12, 6))

        ax = sns.boxplot(
            x="Dataset",
            y=gene,
            hue="Group",
            data=final_df,
            palette={"Control": "#4c72b0", "AD": "#c44e52"},
            showfliers=False,
            width=0.6,
        )
        sns.stripplot(
            x="Dataset",
            y=gene,
            hue="Group",
            data=final_df,
            dodge=True,
            jitter=True,
            color="black",
            alpha=0.3,
            size=3,
        )

        datasets_list = final_df["Dataset"].unique()
        y_range = max(final_df[gene].max() - final_df[gene].min(), 1e-6)
        y_offset = y_range * 0.05

        for i, dataset_name in enumerate(datasets_list):
            subset = final_df[final_df["Dataset"] == dataset_name]
            group_ad = subset[subset["Group"] == "AD"][gene]
            group_ctrl = subset[subset["Group"] == "Control"][gene]

            if len(group_ad) <= 1 or len(group_ctrl) <= 1:
                continue

            _, p_val = stats.ttest_ind(group_ad, group_ctrl)
            sig_symbol = "ns"
            if p_val < 0.001:
                sig_symbol = "***"
            elif p_val < 0.01:
                sig_symbol = "**"
            elif p_val < 0.05:
                sig_symbol = "*"

            x1, x2 = i - 0.2, i + 0.2
            curr_y = subset[gene].max() + y_offset
            plt.plot([x1, x1, x2, x2], [curr_y, curr_y + y_offset, curr_y + y_offset, curr_y], lw=1, c="k")
            plt.text(
                (x1 + x2) * 0.5,
                curr_y + y_offset,
                f"{sig_symbol}\n(p={p_val:.1e})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.title(f"Expression Level of {gene} across Datasets", fontsize=14)
        plt.ylabel("Expression Level")
        plt.xlabel("")

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[:2], labels[:2], title="Group", loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{gene}.png"), dpi=300)
        plt.close()

    print(f"Saved gene expression plots to: {OUTPUT_DIR}")


def main():
    candidate_genes = config.load_candidate_genes()
    print(f"Loaded candidate genes: {len(candidate_genes)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = {}
    for name, ds_config in DATASETS.items():
        df = load_dataset(name, ds_config)
        if df is not None:
            all_data[name] = df
            print(f"  {name}: {df.shape}")
        else:
            print(f"  failed to load {name}")

    plot_gene_boxplots(all_data, candidate_genes)

    print("\n" + "=" * 50)
    print("Gene expression validation finished")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
