# AD-Insight：基于集成特征选择与 Transformer-SVM 对比的阿尔茨海默症诊断研究

语言： [English](README.md) | **简体中文**

About：基于集成特征选择与 Transformer 对比 SVM 的阿尔茨海默症诊断研究 / Alzheimer's diagnosis study based on integrated feature selection and Transformer-vs-SVM comparison.

英文简称：**AD-Insight**

这是一个基于 GEO 基因表达数据的阿尔茨海默症诊断研究流程项目，覆盖从训练集预处理、候选基因筛选、Transformer 建模、SVM 对照训练到外部数据集验证的完整链路。

## 功能概览

- 从原始训练矩阵到外部验证的完整流程。
- 在相同候选基因集下比较 Transformer 与 SVM。
- 支持两个独立 GEO 外部数据集的跨平台验证。
- 提供可选的基因表达图和基因交互解释分析。
- 运行时自动生成 `results/` 目录保存实验输出。

## 仓库结构

```text
.
|-- ad_diagnosis/              # 核心 Python 包
|-- scripts/                   # 命令行入口脚本
|-- docs/                      # 文档与实验说明
|-- requirements.txt
|-- README.md
`-- README_CN.md
```

公开版仓库默认约定：

- 原始数据保存在本地 `data/` 目录，不作为发布仓库内容的一部分。
- 运行结果保存在 `results/` 目录，由程序在运行时自动创建。

## 环境要求

- Python 3.10+
- `pip`
- 可选 CUDA 环境，用于加速 PyTorch 训练

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据目录

数据来源网站：NCBI Gene Expression Omnibus（GEO）。

对应数据集页面：

- `GSE33000`：`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33000`
- `GSE122063`：`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE122063`
- `GSE109887`：`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109887`

请先从 GEO 下载所需文件，再按下面的本地目录结构放入 `data/` 目录。

请在本地准备如下目录结构：

```text
data/
|-- GSE33000mx/
|-- GSE122063yz1/
`-- GSE109887yz2/
```

训练集需要至少包含：

- `data/GSE33000mx/geneMatrix.txt`
- `data/GSE33000mx/clinical.xlsx`

外部验证集目录需要包含：

- `geneMatrix.txt`
- `s1.txt`
- `s2.txt`

补充说明：

- 仓库中使用的本地目录名 `GSE33000mx`、`GSE122063yz1`、`GSE109887yz2` 是项目内部的历史命名。
- 如果你从 GEO 下载后的文件名与项目要求不一致，需要先重命名或整理成项目要求的结构，再运行脚本。

完成预处理后，训练集目录还会生成：

- `data/GSE33000mx/cleaned_gene_matrix.csv`
- `data/GSE33000mx/sample_labels.csv`

更多说明见 `docs/DATA_NOTICE.md`。

## 快速开始

运行完整流程：

```bash
python scripts/run_pipeline.py
```

运行完整流程并生成可选解释性图表：

```bash
python scripts/run_pipeline.py --with-gene-expression --with-gene-interaction
```

分别执行各阶段：

```bash
python scripts/preprocess_data.py
python scripts/run_feature_selection.py
python scripts/train_transformer.py
python scripts/train_svm.py
python scripts/external_validation.py
python scripts/plot_gene_expression.py
python scripts/plot_gene_interaction.py
```

## 流程说明

### 1. 预处理

从训练集原始表达矩阵和临床表中生成清洗后的训练矩阵与二分类标签。

### 2. 特征选择

集成以下 7 种方法进行候选基因排序：

1. T-test
2. Mutual Information
3. XGBoost importance
4. Random Forest importance
5. Elastic Net
6. mRMR
7. Stability Selection

输出写入 `results/feature_selection/`。

### 3. Transformer 训练

训练轻量级 Transformer 分类器，并将模型与相关输出保存到 `results/transformer/`。

### 4. SVM 训练

训练单模型 SVM、Voting SVM、Bagging SVM，并将结果保存到 `results/svm/`。

### 5. 外部验证

在外部 GEO 数据集上评估模型，并将指标和图表保存到 `results/external_validation/`。

### 6. 可选解释分析

- `scripts/plot_gene_expression.py`
- `scripts/plot_gene_interaction.py`

这些脚本会在 `results/gene_validation/` 和 `results/transformer/` 下生成解释性图表。

## 输出目录

程序首次运行时会自动创建如下结构：

```text
results/
|-- feature_selection/
|-- transformer/
|-- svm/
|-- external_validation/
`-- gene_validation/
```

常见输出包括：

- 候选基因排序结果
- 模型参数和预处理器
- ROC 曲线与混淆矩阵
- 外部验证汇总表
- 基因重要性和交互可视化

## 文档

- 英文说明：`README.md`
- 结构说明：`docs/PROJECT_STRUCTURE.md`
- 数据说明：`docs/DATA_NOTICE.md`
- 历史实验摘要：`docs/EXPERIMENT_SUMMARY.md`

## 说明

- 该项目定位为研究代码仓库，用于实验复现和结果分析。
- 当前版本没有自动化测试和 CI。
- 若系统缺少 `SimHei` 字体，中文图表显示效果可能与原环境略有差异。
