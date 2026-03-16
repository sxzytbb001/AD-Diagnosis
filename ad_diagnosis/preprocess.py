import pandas as pd
import numpy as np
import os

from . import config
config.ensure_dirs()

ID_COL = '!Sample_geo_accession'
STATUS_COL = 'Type'


def preprocess_data():
    print(f"当前工作目录: {config.BASE_DIR}")
    print(f"数据目标目录: {config.TRAIN_DATA_DIR}")

    matrix_path = os.path.join(config.TRAIN_DATA_DIR, 'geneMatrix.txt')
    clinical_path = os.path.join(config.TRAIN_DATA_DIR, 'clinical.xlsx')
    output_path = os.path.join(config.TRAIN_DATA_DIR, 'cleaned_gene_matrix.csv')
    labels_path = os.path.join(config.TRAIN_DATA_DIR, 'sample_labels.csv')

    if not os.path.exists(config.TRAIN_DATA_DIR):
        print(f" 错误: 找不到目录 {config.TRAIN_DATA_DIR}")
        return

    print(">>> 步骤1: 读取数据...")
    if not os.path.exists(matrix_path) or not os.path.exists(clinical_path):
        print(f" 错误: 找不到文件。请检查以下路径是否正确:")
        print(f"  - {matrix_path}")
        print(f"  - {clinical_path}")
        return

    print("  正在读取基因矩阵...")
    matrix_gene = pd.read_csv(matrix_path, sep='\t', index_col=0)
    print(f"  ✓ 基因矩阵读取成功: {matrix_gene.shape}")

    clinical = pd.read_excel(clinical_path)

    if ID_COL not in clinical.columns or STATUS_COL not in clinical.columns:
        print(f" 错误: clinical.xlsx 中找不到列 '{ID_COL}' 或 '{STATUS_COL}'")
        return

    print(">>> 步骤2: 筛选样本...")
    clinical[STATUS_COL] = clinical[STATUS_COL].astype(str)

    ad_samples = clinical[clinical[STATUS_COL].str.contains('Alzheimer', case=False, na=False)][ID_COL].tolist()
    control_samples = clinical[clinical[STATUS_COL].str.contains('non-demented', case=False, na=False)][ID_COL].tolist()

    print(f"  - AD 样本数: {len(ad_samples)}")
    print(f"  - Control 样本数: {len(control_samples)}")

    valid_ad = [s for s in ad_samples if s in matrix_gene.columns]
    valid_control = [s for s in control_samples if s in matrix_gene.columns]
    valid_samples = valid_ad + valid_control

    if len(valid_samples) == 0:
        print(" 错误: 有效样本数为0！")
        return

    final_matrix = matrix_gene[valid_samples]

    print(f"  ✓ 最终有效样本: {len(valid_samples)}")
    print(f"  ✓ 最终矩阵形状: {final_matrix.shape}")

    print(f">>> 步骤3: 保存结果至 {config.TRAIN_DATA_DIR} ...")
    final_matrix.to_csv(output_path)

    labels = pd.DataFrame({
        'sample_id': valid_samples,
        'label': ['AD'] * len(valid_ad) + ['Control'] * len(valid_control)
    })
    labels.to_csv(labels_path, index=False)

    print(f"✓ 完成! 清洗后的矩阵: {output_path}")


if __name__ == "__main__":
    preprocess_data()
