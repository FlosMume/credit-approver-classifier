
# credit-approval-classifier

A clean, Git-ready **Windows + WSL** Python project for training **Logistic Regression** and **Decision Tree** (scikit-learn) on a credit-approval dataset (`Approved` = Yes/No). This is a reusable project intended for version control and publishing to GitHub.

## Folder layout
```
credit-approval-classifier/
├─ src/                     # Source code
│  ├─ train.py
│  └─ utils_io.py
├─ data/                    # Put your dataset(s) here (kept out of git)
│  └─ CreditData.csv       # <- save your CSV here (default path)
├─ artifacts/               # Model outputs (ignored by git)
├─ environment.yml          # Conda env spec (recommended)
├─ requirements.txt         # Optional for pip-only users
├─ README.md
└─ .gitignore
```

### About the `scripts/` folder
Originally, helper scripts lived under `scripts/` (PowerShell & Bash runners). Since this project now uses **Conda** as the default workflow and the CLI is simple, the `scripts/` folder is **optional**. You can keep it for one-click runners, but it’s not required. Given your preference for Conda, it’s perfectly fine to remove it.

## Create the Conda environment (recommended)
```bash
conda env create -f environment.yml
conda activate credit-approval-env
```

If you prefer to create it from scratch:
```bash
conda create -n credit-approval-env python=3.11 -y
conda activate credit-approval-env
pip install -U pip
pip install -r requirements.txt
```

> Export later for sharing/repro:
> ```bash
> conda env export --name credit-approval-env > environment.yml
> ```

## Run the project

### Save your data
Place your CSV at:
```
./data/CreditData.csv
```

### Train & generate artifacts
```bash
python src/train.py
```
By default the script reads `./data/CreditData.csv` and writes PNGs/CSVs to `./artifacts/`.
Override paths if needed:
```bash
python src/train.py --csv ./data/Other.csv --out ./artifacts_run2
```

## Headless plotting in WSL/servers
The code forces a **headless** backend (`Agg`) so figures save without a display. To set this at the env level:
```bash
conda activate credit-approval-env
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
printf 'export MPLBACKEND=Agg
' > $CONDA_PREFIX/etc/conda/activate.d/mpl.sh
```

## Design choices explained
- **Pipeline steps as tuples**: The `Pipeline` API requires an ordered list of `(name, estimator)` pairs. The **name** lets you access steps (`pipeline.named_steps['preprocess']`), and the **order** defines execution. Dicts aren’t accepted; tuples keep things explicit and stable.
- **`handle_unknown='ignore'`** (OneHotEncoder): prevents crashes when the test set contains unseen categories by encoding them as all zeros rather than raising an error.
- **`sparse_output=False`** (OneHotEncoder): return a dense array for easier concatenation/inspection. Consider `True` if you have very high-cardinality categoricals and memory pressure.
- **Scaling**: Numeric features are standardized for Logistic Regression (optimizer stability, comparable coefficients), but left unscaled for Decision Trees (scale-invariant).

## Git hygiene
Data and generated artifacts are ignored by `.gitignore`. To keep `data/` in git without committing datasets, a tiny placeholder file is included:
```
data/.gitkeep
```

## Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: credit-approval-classifier (Conda)"
git branch -M main
git remote add origin git@github.com:<your-username>/credit-approval-classifier.git
git push -u origin main
```
