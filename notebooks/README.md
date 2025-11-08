# Credit Approval — Notebook Overview

This folder contains the exploratory notebook that accompanies the `credit-approval-classifier` project.

## 📘 Notebook: `credit_approval_exploratory_analysis.ipynb`

### Purpose
A richly commented, end-to-end demonstration of:
- Data loading (`Approved` = Yes/No)
- Building **Logistic Regression** and **Decision Tree** models with scikit-learn Pipelines
- Explaining parameter choices (`handle_unknown='ignore'`, `sparse_output=False`, tuple-based steps)
- Evaluating model performance (accuracy, precision, recall)
- Extracting and interpreting Logistic Regression coefficients
- Visualizing a Decision Tree to depth=4

### Structure
1. **Imports and design notes** – explains rationale for pipelines and encoders.
2. **Load data** – expects `../data/CreditData.csv` by default.
3. **Split & preprocess** – stratified 80/20 split; type-based transformers.
4. **Model training** – Logistic Regression + Decision Tree.
5. **Evaluation** – confusion matrices, metrics, classification reports.
6. **Coefficient interpretation** – weights & odds ratios for logistic regression.
7. **Tree visualization** – displays top-level decision logic.

### Usage
From this folder:
```bash
conda activate credit-approval-env
jupyter notebook credit_approval_exploratory_analysis.ipynb
```
Ensure your dataset is located at `../data/CreditData.csv`.

### Export options
You can render this notebook as a PDF or Markdown summary:
```bash
jupyter nbconvert --to pdf credit_approval_exploratory_analysis.ipynb
jupyter nbconvert --to markdown credit_approval_exploratory_analysis.ipynb
```

---
**Last updated:** automatically generated for Conda-based project environment.
