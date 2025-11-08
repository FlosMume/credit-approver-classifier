# credit-approval-classifier

A clean, Git-ready **Windows + WSL** Python project for training **Logistic Regression** and **Decision Tree** models (scikit-learn) on a credit-approval dataset (target column: `Approved`, labels `Yes/No`). This is **not** an assignment template; it is a reusable project you can version-control and push to GitHub.

## Why this project?
- Simple, CPU-only scikit-learn baselines (no NVIDIA GPU required).
- Clear, **richly commented** code that explains design choices—especially:
  - Why `Pipeline(steps=[('name', transformer_or_model), ...])` uses **tuples** rather than dicts
  - What `handle_unknown='ignore'` does in `OneHotEncoder`
  - Why `sparse_output=False` is set and when to consider `True`

## Quick start

### Windows PowerShell (venv)
```powershell
cd C:\path\to\credit-approval-classifier
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
python .\src\train.py --csv "C:\path\to\CreditData.csv" --out "artifacts"
```

### WSL (Ubuntu)
```bash
cd ~/credit-approval-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python src/train.py --csv /mnt/c/Users/<you>/Downloads/CreditData.csv --out artifacts
```

### Conda
```bash
conda create -n credit-approval python=3.11 -y
conda activate credit-approval
pip install -r requirements.txt
python src/train.py --csv /path/to/CreditData.csv --out artifacts
```

> **Virtual environment?** Strongly recommended to keep dependencies isolated.

## Do I need an RTX 4070?
**No.** CPU is sufficient; no CUDA or GPU used here.

## Artifacts
- `confusion_matrix_logistic_regression.png`
- `confusion_matrix_decision_tree.png`
- `metrics_summary.csv`
- `logreg_coefficients.csv`
- `logreg_top_positive_coefficients.png` (optional)
- `logreg_top_negative_coefficients.png` (optional)
- `decision_tree.png`

## Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: credit-approval-classifier"
git branch -M main
git remote add origin git@github.com:<your-username>/credit-approval-classifier.git
git push -u origin main
```

## Design choices (high level)
- **Pipeline steps as tuples**: The `Pipeline` API requires an ordered sequence of `(name, estimator)` pairs. The step **name** lets you access/replace steps (`pipeline.named_steps['preprocess']`), and the **order** defines execution. Dicts aren’t accepted by the API, and tuples make the ordering explicit and stable.
- **`handle_unknown='ignore'`** (OneHotEncoder): prevents crashes when the test set contains unseen categories by encoding them as all zeros rather than raising an error.
- **`sparse_output=False`** (OneHotEncoder): return a dense NumPy array for easier concatenation/inspection. Consider sparse output if you have very high-cardinality categoricals and memory pressure.
