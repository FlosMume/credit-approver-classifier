
# Credit Approval Classifier

A concise, Conda-based project demonstrating credit approval prediction using **Logistic Regression** and **Decision Tree** models with scikit-learn.  
This repository includes clean Python scripts for reproducibility and a companion notebook for exploration and interpretation.

---

## 🧭 Folder Layout
```
credit-approval-classifier/
├─ src/                     # Source code
│  ├─ train.py              # Main training script
│  └─ utils_io.py           # CSV loading & I/O helpers
├─ notebooks/               # Exploratory & training notebooks
│  ├─ credit_approval_exploratory_analysis.ipynb
│  └─ README.md             # Notebook overview
├─ data/                    # Local datasets (ignored by git)
│  └─ .gitkeep
├─ artifacts/               # Model outputs (ignored by git)
├─ environment.yml          # Conda environment definition
├─ requirements.txt         # Optional pip requirements
├─ README.md                # Project documentation (this file)
└─ .gitignore
```

---

## ⚙️ Environment Setup

### Create and activate the Conda environment
```bash
conda env create -f environment.yml
conda activate credit-approval-env
```

### (Alternative manual setup)
```bash
conda create -n credit-approval-env python=3.11 -y
conda activate credit-approval-env
pip install -U pip
pip install -r requirements.txt
```

---

## 🚀 Run the Project

1. Place your dataset at `./data/CreditData.csv`
2. Train and generate outputs:
   ```bash
   python src/train.py
   ```
3. All artifacts (plots, metrics, coefficients) will be saved under `./artifacts/`

---

## 📘 Notebook Viewing

The exploratory notebook is available under `notebooks/credit_approval_exploratory_analysis.ipynb`.

Open it directly in **VS Code**, **JupyterLab**, or **GitHub** for a full, commented walkthrough — no PDF conversion required.

A concise summary of the notebook is also available in `notebooks/README.md`.

---

## 🧹 Git Hygiene

- The `data/` and `artifacts/` folders are ignored by git to keep the repo lightweight.  
- A `.gitkeep` placeholder ensures that `data/` remains visible for new collaborators.

---

## 🏁 Summary

- **Models:** Logistic Regression, Decision Tree  
- **Tools:** scikit-learn, pandas, numpy, matplotlib  
- **Goal:** Predict credit approval and interpret model coefficients  
- **Environment:** Conda (`credit-approval-env`)

Clean, reproducible, and ready for further experimentation.
