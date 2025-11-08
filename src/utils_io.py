import os
import pandas as pd

def load_credit_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'Approved' not in df.columns:
        raise ValueError(f"'Approved' column not found. Columns are: {df.columns.tolist()}")
    return df

def ensure_outdir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.abspath(out_dir)
