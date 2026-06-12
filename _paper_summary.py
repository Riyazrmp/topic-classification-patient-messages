"""Build paper-ready summary tables from results/per_corpus_*.csv.

Writes:
  - results/paper_table_macro_by_corpus.md       Markdown table (paste straight into draft)
  - results/paper_table_safety_critical.md       Per-label F1 for Urgent/Emergent/Service Complaint
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
RES = ROOT / "results"

MODEL_ORDER = ["Ensemble", "BERT", "RandomForest", "XGBoost", "SVM", "LSTM"]
SAFETY_LABELS = ["Emergent", "Urgent", "Service Complaint"]


def fmt(v, n=2):
    try:
        return f"{float(v):.{n}f}"
    except Exception:
        return "—"


def main():
    macro = pd.read_csv(RES / "per_corpus_macro_f1.csv")
    perlbl = pd.read_csv(RES / "per_corpus_per_label_f1.csv")

    # Macro F1 by corpus, wide format
    pivot = macro.pivot(index="Model", columns="Corpus", values=["MacroF1", "CI_lo", "CI_hi", "N"])
    rows = []
    for m in MODEL_ORDER:
        if m not in pivot.index:
            continue
        cells = []
        for corpus in ["Combined", "Rwanda", "Canada"]:
            f1 = pivot.loc[m, ("MacroF1", corpus)]
            lo = pivot.loc[m, ("CI_lo", corpus)]
            hi = pivot.loc[m, ("CI_hi", corpus)]
            n  = pivot.loc[m, ("N", corpus)]
            cells.append(f"{fmt(f1)} [{fmt(lo)}, {fmt(hi)}] (n={int(n)})")
        rows.append([m] + cells)

    lines = [
        "| Model | Combined | Rwanda | Canada |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    (RES / "paper_table_macro_by_corpus.md").write_text("\n".join(lines) + "\n")
    print(" wrote results/paper_table_macro_by_corpus.md")

    # Safety-critical labels: F1 per model per corpus
    safe = perlbl[perlbl["Label"].isin(SAFETY_LABELS)].copy()
    safe = safe.pivot_table(index="Model", columns=["Corpus", "Label"], values="F1")
    lines2 = ["Safety-critical per-label F1 (pooled / Rwanda / Canada):", ""]
    lines2.append("| Model | " + " | ".join([f"{l} ({c})" for c in ["Combined", "Rwanda", "Canada"] for l in SAFETY_LABELS]) + " |")
    lines2.append("|---" + "|---" * (3 * len(SAFETY_LABELS)) + "|")
    for m in MODEL_ORDER:
        if m not in safe.index:
            continue
        row = [m]
        for c in ["Combined", "Rwanda", "Canada"]:
            for l in SAFETY_LABELS:
                try:
                    row.append(fmt(safe.loc[m, (c, l)]))
                except KeyError:
                    row.append("—")
        lines2.append("| " + " | ".join(row) + " |")
    (RES / "paper_table_safety_critical.md").write_text("\n".join(lines2) + "\n")
    print(" wrote results/paper_table_safety_critical.md")

    print()
    print((RES / "paper_table_macro_by_corpus.md").read_text())
    print()
    print((RES / "paper_table_safety_critical.md").read_text())


if __name__ == "__main__":
    main()
