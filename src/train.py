#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # or: os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt

from utils_io import load_credit_csv, ensure_outdir

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score,
    classification_report
)

def build_pipelines(categorical_cols, numeric_cols):
    # OneHotEncoder config with rich comments:
    # - handle_unknown='ignore':
    #   Without this, an unseen category in test data throws an error.
    #   With 'ignore', we encode that unseen category as all zeros for that feature block,
    #   which keeps inference robust.
    # - sparse_output=False:
    #   Return dense arrays so we can easily concatenate with dense scaled numerics and
    #   inspect intermediate results. If you face memory pressure, consider True.
    cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Scale numerics for Logistic Regression; trees don't need scaling
    num_scaled = StandardScaler()
    num_passthrough = 'passthrough'

    preprocess_logreg = ColumnTransformer(
        transformers=[
            ('cat', cat, categorical_cols),
            ('num', num_scaled, numeric_cols),
        ]
    )

    preprocess_tree = ColumnTransformer(
        transformers=[
            ('cat', cat, categorical_cols),
            ('num', num_passthrough, numeric_cols),
        ]
    )

    # Why Pipeline steps are tuples:
    #   Pipeline requires an ordered list of (name, estimator) to define **both** the
    #   execution order and the stable names for addressing steps. Dicts are not accepted.
    logreg = Pipeline(steps=[
        ('preprocess', preprocess_logreg),
        ('model', LogisticRegression(max_iter=1000))  # higher max_iter for convergence safety
    ])

    tree = Pipeline(steps=[
        ('preprocess', preprocess_tree),
        ('model', DecisionTreeClassifier(random_state=42))
    ])

    return logreg, tree

def evaluate_and_plot(name, pipeline, X_test, y_test, out_dir, positive_label='Yes'):
    y_pred = pipeline.predict(X_test)
    other = [lab for lab in sorted(y_test.unique()) if lab != positive_label]
    neg_label = other[0] if other else 'No'
    labels = [positive_label, neg_label]

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=positive_label, zero_division=0)

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f} (pos='{positive_label}')")
    print(f"Recall   : {rec:.4f} (pos='{positive_label}')\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    plt.title(f'{name} — Confusion Matrix')
    fig.tight_layout()
    stub = name.lower().replace(' ', '_')
    fig.savefig(os.path.join(out_dir, f'confusion_matrix_{stub}.png'), dpi=200)
    plt.close(fig)

    return {'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'cm': cm.tolist()}

def main():
    parser = argparse.ArgumentParser(description="credit-approval-classifier")
    parser.add_argument(
        "--csv",
        default=os.path.join("data", "CreditData.csv"),
        help="Path to the CSV file (default: ./data/CreditData.csv)"
    )
    parser.add_argument("--out", default="artifacts", help="Output directory for images/CSV")
    parser.add_argument("--max_depth", type=int, default=4, help="Max depth to show in decision_tree.png")
    parser.add_argument("--no_coef_charts", action="store_true", help="Skip coefficient bar charts")
    args = parser.parse_args()

    out_dir = ensure_outdir(args.out)
    df = load_credit_csv(args.csv)

    TARGET = 'Approved'
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print("Categorical:", categorical_cols)
    print("Numeric    :", numeric_cols)
    print("\nTarget distribution:\n", y.value_counts(dropna=False))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logreg, tree = build_pipelines(categorical_cols, numeric_cols)
    logreg.fit(X_train, y_train)
    tree.fit(X_train, y_train)

    m_logreg = evaluate_and_plot("Logistic Regression", logreg, X_test, y_test, out_dir, positive_label='Yes')
    m_tree   = evaluate_and_plot("Decision Tree",        tree,   X_test, y_test, out_dir, positive_label='Yes')

    metrics_df = pd.DataFrame([
        {k: v for k, v in m_logreg.items() if k != 'cm'},
        {k: v for k, v in m_tree.items() if k != 'cm'},
    ])
    metrics_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    # Logistic Regression coefficients
    ohe = logreg.named_steps['preprocess'].named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(ohe.feature_names_in_).tolist()
    num_feature_names = numeric_cols
    all_features = cat_feature_names + num_feature_names

    coef = logreg.named_steps['model'].coef_.ravel()
    coef_df = pd.DataFrame({"feature": all_features, "coefficient": coef}).sort_values("coefficient", ascending=False)
    coef_df.to_csv(os.path.join(out_dir, "logreg_coefficients.csv"), index=False)

    if not args.no_coef_charts:
        top_n = 15
        top_pos = coef_df.head(top_n)
        fig = plt.figure(figsize=(8,6))
        plt.barh(top_pos["feature"], top_pos["coefficient"])
        plt.title("Logistic Regression — Top Positive Coefficients")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "logreg_top_positive_coefficients.png"), dpi=200)
        plt.close(fig)

        top_neg = coef_df.tail(top_n).sort_values("coefficient")
        fig = plt.figure(figsize=(8,6))
        plt.barh(top_neg["feature"], top_neg["coefficient"])
        plt.title("Logistic Regression — Top Negative Coefficients")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "logreg_top_negative_coefficients.png"), dpi=200)
        plt.close(fig)

    # Decision tree visualization
    ct = tree.named_steps['preprocess']
    Xt_train = ct.fit_transform(X_train)
    feature_names = ct.get_feature_names_out().tolist()

    plain_tree = DecisionTreeClassifier(random_state=42)
    plain_tree.fit(Xt_train, y_train)

    fig = plt.figure(figsize=(20, 12))
    plot_tree(
        plain_tree,
        feature_names=feature_names,
        class_names=sorted(y_train.unique()),
        filled=False, rounded=True, proportion=True, max_depth=args.max_depth
    )
    plt.title(f"Decision Tree (visualized to depth={args.max_depth})")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "decision_tree.png"), dpi=220)
    plt.close(fig)

    print(f"\nAll done. Artifacts saved to: {out_dir}")

if __name__ == "__main__":
    main()
