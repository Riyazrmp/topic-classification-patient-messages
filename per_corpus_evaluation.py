"""
Per-corpus and per-label evaluation for all six classifiers, with bootstrap CIs.

Inputs (predictions/ only — no raw data files required):
  - predictions/<Model>_preds.npz       (y_true (584,19) int, y_pred (584,19) numeric)
  - predictions/Ensemble_test_predictions.json
      (sanitized: each record has "true_labels", "predicted_labels",
       "source_corpus" — no conversation text)

Outputs (results/):
  - per_corpus_macro_f1.csv            wide table: Model x {Combined, Rwanda, Canada} with 95% CI
  - per_corpus_per_label_f1.csv        long table: Model, Corpus, Label, F1, support
  - bootstrap_macro_f1.csv             1,000-resample bootstrap distributions per Model x Corpus

Run:  .venv/bin/python per_corpus_evaluation.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SEED = 42
N_BOOTSTRAP = 1000
ROOT = Path(__file__).parent
PRED_DIR = ROOT / "predictions"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_predictions():
    """Load all saved predictions, returning {model: (y_true, y_pred)}."""
    out = {}
    for name in ["LSTM", "RandomForest", "XGBoost", "SVM", "BERT"]:
        p = PRED_DIR / f"{name}_preds.npz"
        if p.exists():
            d = np.load(p)
            y_true = d["y_true"].astype(int)
            y_pred = d["y_pred"]
            if y_pred.dtype != int:
                y_pred = (y_pred > 0.5).astype(int) if y_pred.max() <= 1 else y_pred.astype(int)
            out[name] = (y_true, y_pred)

    ens_path = PRED_DIR / "Ensemble_test_predictions.json"
    if ens_path.exists():
        ens = json.load(open(ens_path))
        labels = list(ens[0]["true_labels"].keys())
        y_true = np.array([[r["true_labels"][l] for l in labels] for r in ens], dtype=int)
        y_pred = np.array([[r["predicted_labels"][l] for l in labels] for r in ens], dtype=int)
        out["Ensemble"] = (y_true, y_pred)
        out["_labels_from_ensemble"] = labels
        out["_source_from_ensemble"] = np.array([r["source_corpus"] for r in ens])
    return out


def bootstrap_macro_f1(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return np.nan, (np.nan, np.nan), np.array([])
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[i] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    point = f1_score(y_true, y_pred, average="macro", zero_division=0)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return point, (lo, hi), scores


def per_label_f1(y_true, y_pred, labels):
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = y_true.sum(axis=0).astype(int)
    return list(zip(labels, f1, support))


def main():
    preds = load_predictions()
    labels_ens = preds.pop("_labels_from_ensemble", None)
    src_ens = preds.pop("_source_from_ensemble", None)

    if src_ens is None or labels_ens is None:
        raise RuntimeError("Ensemble JSON missing or malformed; need its true_labels/source_corpus fields for per-corpus eval.")

    label_keep = labels_ens
    src_counts = pd.Series(src_ens).value_counts().to_dict()
    print(f"Test set: {len(src_ens)} docs (Rwanda + Canada)")
    print(f"Labels kept: {len(label_keep)}")
    print(f"Source counts in test set (per Ensemble JSON): {src_counts}")

    macro_rows = []
    perlabel_rows = []
    boot_rows = []

    for model_name, (y_true, y_pred) in preds.items():
        if y_true.shape[0] != len(src_ens):
            print(f"  {model_name}: shape mismatch ({y_true.shape} vs source {len(src_ens)}), skipping")
            continue
        for corpus in ["Combined", "Rwanda", "Canada"]:
            if corpus == "Combined":
                mask = np.ones(len(src_ens), dtype=bool)
            else:
                mask = src_ens == corpus
            yt = y_true[mask]
            yp = y_pred[mask]
            point, (lo, hi), scores = bootstrap_macro_f1(yt, yp)
            macro_rows.append({
                "Model": model_name,
                "Corpus": corpus,
                "N": int(mask.sum()),
                "MacroF1": round(point, 4),
                "CI_lo": round(lo, 4),
                "CI_hi": round(hi, 4),
            })
            print(f"  {model_name:14s} {corpus:8s} n={int(mask.sum()):4d}  Macro F1 {point:.4f}  [{lo:.4f}, {hi:.4f}]")
            for label, f1, sup in per_label_f1(yt, yp, label_keep):
                perlabel_rows.append({
                    "Model": model_name,
                    "Corpus": corpus,
                    "Label": label,
                    "F1": round(float(f1), 4),
                    "Support": int(sup),
                })
            for s in scores:
                boot_rows.append({"Model": model_name, "Corpus": corpus, "MacroF1": float(s)})
        print()

    pd.DataFrame(macro_rows).to_csv(RESULTS_DIR / "per_corpus_macro_f1.csv", index=False)
    pd.DataFrame(perlabel_rows).to_csv(RESULTS_DIR / "per_corpus_per_label_f1.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(RESULTS_DIR / "bootstrap_macro_f1.csv", index=False)
    print(f"\nWrote results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
