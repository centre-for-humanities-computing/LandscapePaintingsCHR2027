"""
Trains a classifier on embeddings loaded from a local parquet file (in
./data/).

Produces fold-validated predictions/metrics only, but saves both the
per-fold models and a final model (trained on all data) in
classifier.joblib.

predictions.parquet columns:
    p_true, p_wrong, margin, predicted_class, misclassified, fold_id
"""

import os
from datetime import datetime
 
import numpy as np
import pandas as pd
import polars as pl
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
 

# ── CONFIG ────────────────────────────────────────────────────────────────────

CONFIG = {
    "data": {
        "local_path": "data/classification.parquet",   # <-- set to your actual filename
    },
    "class_col": "class",
    "embedding_col":            "dinov2-base-backbone-embedding",
    # available columns: 'eva-02-backbone-embedding', 'eva-02-head-embedding', 'eva-02-backbone-embedding-cls', 
    # 'dinov2-base-backbone-embedding', 'dinov2-base-head-embedding',
    # 'dinov2-giant-backbone-embedding', 'dinov2-giant-head-embedding', 
    # 'siglip-SO400M-backbone-embedding', 'siglip-SO400M-head-embedding', 'colour_histogram'
    "min_paintings_per_artist": 10,
    "n_splits":                 5,
}

_out_dir = os.path.join("classifier", CONFIG["embedding_col"])
CONFIG["output"] = {
    "dir":          _out_dir,
    "predictions":  os.path.join(_out_dir, "predictions.parquet"),
    "classifier":   os.path.join(_out_dir, "classifier.joblib"),
    "metrics_txt":  os.path.join(_out_dir, "metrics_report.txt"),
}

# ── Utilities ────────────────────────────────────────────────────────────────────

def load_parquet_robust(path):
    return pl.read_parquet(path).to_pandas()

def make_classifier():
    """Single source of truth for the model used per-fold and for the
    final model -- change hyperparameters here once."""
    return CalibratedClassifierCV(
        LogisticRegression(max_iter=5000, class_weight="balanced"),
        cv=3, method="sigmoid",
    )


def create_artist_splits(df, n_splits=5):
    """GroupKFold-by-artist, done independently per class then combined,
    so fold membership is balanced across classes.

    GroupKFold.split() returns POSITIONS (0..len-1), not df index values --
    since each class is split on its own subset, positions must be mapped
    back via cls_indices[...] before combining across classes, or the two
    classes' position arrays collide when concatenated."""
    gkf = GroupKFold(n_splits=n_splits)
    per_class_splits = []

    for cls in sorted(df["main_class"].unique()):
        cls_df      = df[df["main_class"] == cls]
        cls_indices = cls_df.index.values
        cls_artists = cls_df["artist"].values
        local_splits = list(gkf.split(cls_indices, groups=cls_artists))
        per_class_splits.append([
            (cls_indices[tr], cls_indices[te]) for tr, te in local_splits
        ])

    return [
        (np.concatenate([s[i][0] for s in per_class_splits]),
         np.concatenate([s[i][1] for s in per_class_splits]))
        for i in range(n_splits)
    ]


def format_metrics_report(embedding_col, class_names, fold_metrics, y_true, y_pred):
    """Aggregate (all OOF predictions concatenated) + per-fold mean ± std."""
    lines = []
    lines.append("=" * 66)
    lines.append(f"Classification Report — {embedding_col}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 66)

    lines.append("\n── Aggregate (all folds concatenated) ─────────────────────────\n")
    lines.append(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    lines.append("── Per-fold mean ± std ─────────────────────────────────────────\n")
    lines.append(f"{'Metric':<35}  {'Mean':>8}  {'Std':>8}")
    lines.append("-" * 55)

    metric_keys = [
        (f"precision_{class_names[0]}",  f"Precision  ({class_names[0]})"),
        (f"recall_{class_names[0]}",     f"Recall     ({class_names[0]})"),
        (f"f1_{class_names[0]}",         f"F1-score   ({class_names[0]})"),
        (f"precision_{class_names[1]}",  f"Precision  ({class_names[1]})"),
        (f"recall_{class_names[1]}",     f"Recall     ({class_names[1]})"),
        (f"f1_{class_names[1]}",         f"F1-score   ({class_names[1]})"),
        ("accuracy",                     "Accuracy"),
    ]
    for key, label in metric_keys:
        vals = np.array([m[key] for m in fold_metrics])
        lines.append(f"  {label:<33}  {vals.mean():>8.4f}  {vals.std():>8.4f}")

    lines.append("")
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    embedding_col = CONFIG["embedding_col"]
    class_col = CONFIG["class_col"]

    print(f"[info] Loading: {CONFIG['data']['local_path']}")
    df = load_parquet_robust(CONFIG["data"]["local_path"])
    n_total = len(df)

    df[class_col] = df[class_col].astype(str).str.strip().str.lower()
    df = df[df[class_col].isin(["danish", "norwegian"])].reset_index(drop=True)
    print(f"[info] Kept {len(df)}/{n_total} row(s) with a valid '{class_col}' label "
          f"(danish/norwegian) -- {n_total - len(df)} unlabeled/other dropped.")
    df = df.rename(columns={class_col: "main_class"})

    n_artists_before = df["artist"].nunique()
    painting_counts = df.groupby("artist")["main_class"].transform("count")
    df = df[painting_counts >= CONFIG["min_paintings_per_artist"]].reset_index(drop=True)
    print(f"Kept {df['artist'].nunique()} artists ({len(df)} paintings) after filtering "
          f"(removed {n_artists_before - df['artist'].nunique()} with too few paintings).\n")

    X  = np.vstack(df[embedding_col].values)
    le = LabelEncoder()
    y  = le.fit_transform(df["main_class"])
    class_names = list(le.classes_)   # ['danish', 'norwegian']

    splits = create_artist_splits(df, n_splits=CONFIG["n_splits"])

    # Every row must appear in EXACTLY one fold's test set -- cheap check,
    # catches a real index-collision bug class this project hit before.
    all_test_idx = np.concatenate([test_idx for _, test_idx in splits])
    assert len(all_test_idx) == len(df) == len(set(all_test_idx.tolist())), \
        "Fold test sets don't exactly cover df's rows once each -- index bug."
    assert set(all_test_idx.tolist()) == set(df.index.tolist()), \
        "Fold test sets cover the wrong rows -- index bug."

    probs       = np.zeros((len(X), 2))
    fold_metrics = []
    fold_models = []
    fold_id_arr = np.full(len(X), -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        clf = make_classifier()
        clf.fit(X[train_idx], y[train_idx])
        probs[test_idx] = clf.predict_proba(X[test_idx])
        fold_models.append(clf)
        fold_id_arr[test_idx] = fold_i

        fold_preds = probs[test_idx].argmax(axis=1)
        p, r, f, _ = precision_recall_fscore_support(
            y[test_idx], fold_preds, labels=[0, 1], zero_division=0
        )
        acc = (fold_preds == y[test_idx]).mean()
        fold_metrics.append({
            f"precision_{class_names[0]}": p[0], f"recall_{class_names[0]}": r[0], f"f1_{class_names[0]}": f[0],
            f"precision_{class_names[1]}": p[1], f"recall_{class_names[1]}": r[1], f"f1_{class_names[1]}": f[1],
            "accuracy": acc,
        })
        print(f"Fold {fold_i + 1}/{CONFIG['n_splits']} -- acc: {acc:.4f}")

    df["fold_id"] = fold_id_arr

    # Final model, trained on ALL data. Saved for ablation_local_unified.py's
    # "full" mode -- not evaluated here.
    final_clf = make_classifier()
    final_clf.fit(X, y)

    df["p_true"]          = probs[np.arange(len(y)), y]
    df["p_wrong"]         = probs[np.arange(len(y)), 1 - y]
    df["margin"]          = df["p_true"] - df["p_wrong"]
    df["predicted_class"] = le.inverse_transform(probs.argmax(axis=1))
    df["misclassified"]   = df["predicted_class"] != df["main_class"]

    df.drop(columns=["image"]).to_parquet(CONFIG["output"]["predictions"])

    report_txt = format_metrics_report(
        embedding_col, class_names, fold_metrics, df["main_class"], df["predicted_class"],
    )
    print(report_txt)
    with open(CONFIG["output"]["metrics_txt"], "w", encoding="utf-8") as fh:
        fh.write(report_txt)
    print(f"Metrics saved -> {CONFIG['output']['metrics_txt']}")

    dump(
        {
            "model": final_clf,
            "fold_models": fold_models,
            "label_encoder": le,
            "embedding_col": embedding_col,
        },
        CONFIG["output"]["classifier"],
    )
    print("Training complete")


if __name__ == "__main__":
    os.makedirs(CONFIG["output"]["dir"], exist_ok=True)
    main()