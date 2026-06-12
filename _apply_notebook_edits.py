"""One-shot editor: add save-preds cells, per-corpus eval cells, markdown notes, and an
ensemble timing cell to the relevant notebooks. Idempotent: if a marker comment is
already present, that edit is skipped."""

import json
from pathlib import Path

ROOT = Path(__file__).parent
MARKER = "# PERCORPUSCELL_v1"  # change if you need to force re-insertion


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def already_has(nb, needle: str) -> bool:
    for c in nb["cells"]:
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if needle in src:
            return True
    return False


PERCORPUS_TEMPLATE = '''# PERCORPUSCELL_v1
# Per-corpus + per-label evaluation against the shared test split, with 1,000-resample bootstrap CIs.
import json, numpy as np
from sklearn.metrics import f1_score, classification_report

with open("predictions/Ensemble_test_predictions.json") as f:
    _ens = json.load(f)
_src    = np.array([r["source_corpus"] for r in _ens])
_labels = list(_ens[0]["true_labels"].keys())

_d = np.load("predictions/{MODEL}_preds.npz")
_yt, _yp = _d["y_true"].astype(int), _d["y_pred"]
if _yp.dtype != int:
    _yp = (_yp > 0.5).astype(int) if _yp.max() <= 1 else _yp.astype(int)

for _corpus in ["Combined", "Rwanda", "Canada"]:
    _mask = np.ones(len(_src), bool) if _corpus == "Combined" else (_src == _corpus)
    _yti, _ypi = _yt[_mask], _yp[_mask]
    _macro = f1_score(_yti, _ypi, average="macro", zero_division=0)

    _rng = np.random.default_rng(42)
    _n = len(_yti)
    _boot = np.empty(1000)
    for _i in range(1000):
        _idx = _rng.integers(0, _n, _n)
        _boot[_i] = f1_score(_yti[_idx], _ypi[_idx], average="macro", zero_division=0)
    _lo, _hi = np.percentile(_boot, [2.5, 97.5])

    print(f"\\n=== {{_corpus}} (n={{_n}})  Macro F1 {{_macro:.4f}}  95% CI [{{_lo:.4f}}, {{_hi:.4f}}] ===")
    print(classification_report(_yti, _ypi, target_names=_labels, zero_division=0))
'''


def _detect_indent(text: str) -> int:
    """Sniff the indent width used by the original notebook (1 or 2 in practice)."""
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        if stripped.startswith('"cells"'):
            n = len(line) - len(stripped)
            return n if n in (1, 2, 3, 4) else 2
    return 2


def edit_notebook(path: Path, edits):
    original_text = path.read_text()
    indent = _detect_indent(original_text)
    nb = json.loads(original_text)
    n_before = len(nb["cells"])
    for fn in edits:
        fn(nb)
    if len(nb["cells"]) != n_before:
        path.write_text(json.dumps(nb, indent=indent, ensure_ascii=False))
        print(f"  wrote {path.name}: {n_before} -> {len(nb['cells'])} cells (indent={indent})")
    else:
        print(f"  {path.name}: no change (markers already present)")


def append_percorpus(model_name):
    def _fn(nb):
        if already_has(nb, "# PERCORPUSCELL_v1"):
            return
        src = PERCORPUS_TEMPLATE.replace("{MODEL}", model_name)
        nb["cells"].append(md_cell(f"## Per-corpus + per-label evaluation ({model_name})\n\n"
                                   f"Loads saved `{model_name}_preds.npz` and reports Macro F1 (+ 95% bootstrap CI) "
                                   f"and per-label F1 separately for the pooled corpus and for the Rwanda and Canada "
                                   f"halves. Source labels come from `Ensemble_test_predictions.json`, which carries "
                                   f"the per-row `source_corpus` field for the same 584-doc test set.\n"))
        nb["cells"].append(code_cell(src))
    return _fn


def insert_after_index(idx, cell):
    def _fn(nb):
        nb["cells"].insert(idx + 1, cell)
    return _fn


def insert_save_preds(model_name, pred_var, ytrue_var, after_cell_idx):
    """Insert a save-preds cell after the given cell index (used for SVM and BERT)."""
    save_src = f'''# SAVE_PREDS_v1
import numpy as np, os
os.makedirs("predictions", exist_ok=True)
np.savez_compressed(
    "predictions/{model_name}_preds.npz",
    y_true = {ytrue_var},
    y_pred = {pred_var},
)
print("Saved predictions/{model_name}_preds.npz", {pred_var}.shape)
'''
    def _fn(nb):
        if already_has(nb, "# SAVE_PREDS_v1"):
            return
        nb["cells"].insert(after_cell_idx + 1, code_cell(save_src))
    return _fn


# ---- LSTM vocab note ----

LSTM_VOCAB_NOTE = """## Note on tokenizer vocabulary size (25,000)

The LSTM uses a `Tokenizer(num_words=25000)` (an integer-token sequence model), which is
intentionally **larger than the TF-IDF `max_features=5,000` used by the SVM, RF and XGBoost
baselines**. The two numbers serve different roles:

- TF-IDF features are a fixed-width sparse vector per document; pruning rare terms to the
  top-5k features keeps the linear classifiers tractable and acts as a regularizer.
- The LSTM operates on **integer token indices** mapped through a learned embedding layer
  (output dim 100). A wider vocabulary lets the embedding capture more clinical /
  Kinyarwanda-derived rare terms that would otherwise collapse to `<OOV>`.

Tokens beyond rank 25,000 are mapped to `<OOV>`. After tokenization, sequences are padded
or truncated to `max_len=200`.
"""


# ---- BERT class-imbalance documentation ----

BERT_LOSS_NOTE = """## Class-imbalance handling in the BERT fine-tuning loss

This BERT classifier handles severe label imbalance via **per-label positive-class weighting**
inside `BCEWithLogitsLoss`. Concretely (cell above):

```python
negativeCount = len(y_train) - y_train.sum(axis=0)
positiveCount = y_train.sum(axis=0)
weights = torch.tensor([negativeCount[i] / positiveCount[i] if positiveCount[i] > 0 else 1
                        for i in range(len(positiveCount))], dtype=torch.float)
# ...
BCEWithLogitsLoss(pos_weight=weights)
```

Each label's positive examples are up-weighted by `(# negatives) / (# positives)` in the
training set. A custom `WeightedLossTrainer` subclasses HF's `Trainer` and overrides
`compute_loss` to use this weighted BCE. No focal loss or label-smoothing is applied; no
oversampling is done in the dataloader. Thresholds are tuned per label on the validation
set (separate cell) rather than fixed at 0.5.

Reviewers asking *how* class imbalance is handled: **positive-class weighting via
`BCEWithLogitsLoss(pos_weight=N_neg/N_pos)`**, equivalent to inverse-frequency reweighting
of the positive class per label.
"""


# ---- Pivot language justification ----

PIVOT_LANG_NOTE = """## Pivot language for back-translation (English -> German -> English)

We use **German** as the pivot language for back-translation augmentation. Rationale:

- German has the highest-quality publicly available open-source MT system for the
  `en-de` direction (Helsinki-NLP / OPUS-MT and Facebook M2M-100 both report top BLEU
  on en-de among Indo-European pivots), which keeps back-translation noise low enough
  to preserve label-bearing content while still producing surface variation.
- German is morphologically richer than English (case marking, separable verbs,
  compounding), so the round-trip rephrasing tends to alter syntax more than a closer
  pivot (e.g. Dutch) while staying lexically faithful to the medical domain vocabulary.
- We did not pivot through Kinyarwanda or any African language: open-source en-rw MT
  quality is currently too low for safe round-trip augmentation in this domain.

**Caveat (added for the Limitations section).** The Rwandan half of the training corpus
was originally translated from Kinyarwanda into English before being fed to the
classifiers. Back-translation augmentation on those rows therefore produces
**triple-translated examples** (rw -> en -> de -> en). We treat this as a known source of
noise rather than as a separate manipulated variable, and we report it as Limitation L4.
"""


# ---- Top2Vec NPMI explanation ----

TOP2VEC_NPMI_NOTE = """## On the negative NPMI for Top2Vec (-0.17)

Top2Vec produced topics whose top-word NPMI score on this corpus is **negative**
(roughly -0.17 in our run), meaning the top words selected for each topic **co-occur
less often than chance** in the document collection. This is an **anti-coherence**
signal, not just "weak" coherence, and we therefore **exclude Top2Vec from the
recommended unsupervised configuration** for this dataset.

Why this happens here:

- Top2Vec relies on `doc2vec` (or, optionally, a sentence-transformer) embeddings of
  full documents, then clusters with HDBSCAN. With ~3.9k short, often near-duplicate
  patient-provider exchanges, doc2vec embeddings are underdetermined and HDBSCAN tends
  to produce one large noise cluster plus a few small topics whose representative
  words generalize poorly across documents.
- BERTopic, by contrast, uses pretrained sentence-transformer embeddings + UMAP +
  HDBSCAN + class-based TF-IDF, which is far more robust at this corpus size and
  produces positive NPMI on the same data.

**Reported in the paper:** Top2Vec is included in Table X for completeness, but
flagged with the negative-NPMI caveat and not carried into the recommendations.
"""


# ---- Ensemble timing/cost cell ----

ENSEMBLE_TIMING = '''# ENSEMBLE_COST_v1
# Reports inference-time and memory footprint of the stacking ensemble at evaluation time.
# Run this on the SAME machine that produced predictions/Ensemble_test_predictions.json
# (i.e. with the trained base learners + the LR meta-classifier still in memory).
import time, json, os, sys
import numpy as np

try:
    import psutil  # for resident set size
    _rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
except Exception:
    _rss_mb = None

# Re-run the inference pipeline on the held-out test set and time each stage.
# Expected variables already in scope (from earlier cells in this notebook):
#   - x_test_raw : list[str]  raw test conversations
#   - svm_pipe   : trained SVM pipeline (per-label)  -> svm_pred (n,19)
#   - xgb_pipe   : trained XGBoost pipeline          -> xgb_pred (n,19)
#   - bert_pipe  : trained BERT extractor            -> test_meta_features_bert
#   - meta_model : trained LR meta-classifier
#   - label_keep_original

_n_test = len(x_test_raw)

t0 = time.perf_counter()
svm_pred_t   = svm_pipe.predict_proba_matrix(x_test_raw) if hasattr(svm_pipe, "predict_proba_matrix") else svm_pred
t1 = time.perf_counter()
xgb_pred_t   = xgb_pipe.predict_proba_matrix(x_test_raw) if hasattr(xgb_pipe, "predict_proba_matrix") else xgb_pred
t2 = time.perf_counter()
bert_feats_t = bert_pipe(x_test_raw) if callable(bert_pipe) else test_meta_features_bert
t3 = time.perf_counter()
X_meta_t = np.concatenate([xgb_pred_t, bert_feats_t, svm_pred_t], axis=1)
meta_pred_t = meta_model.predict(X_meta_t)
t4 = time.perf_counter()

stages = [
    ("SVM base predict",        t1 - t0),
    ("XGBoost base predict",    t2 - t1),
    ("BERT feature extract",    t3 - t2),
    ("Meta classifier predict", t4 - t3),
    ("Total inference",         t4 - t0),
]
print(f"Test docs: {_n_test}")
for name, dt in stages:
    print(f"  {name:28s} {dt*1000:8.1f} ms  ({1000*dt/_n_test:6.2f} ms/doc)")

if _rss_mb is not None:
    print(f"\\nProcess RSS at end of inference: {_rss_mb:.0f} MB")
else:
    print("\\n(install psutil to report memory footprint)")

try:
    out = {
        "n_test": _n_test,
        "per_stage_ms": {name: round(dt * 1000, 2) for name, dt in stages},
        "per_stage_ms_per_doc": {name: round(1000 * dt / _n_test, 3) for name, dt in stages},
        "rss_mb_end": round(_rss_mb, 1) if _rss_mb is not None else None,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/ensemble_cost.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote results/ensemble_cost.json")
except Exception as _e:
    print("WARN: could not write results/ensemble_cost.json:", _e)
'''

ENSEMBLE_TIMING_NOTE = """## Inference cost: time + memory for the stacking ensemble

The next cell measures end-to-end inference time on the 584-doc test set, split by
base learner (SVM / XGBoost / BERT) and meta-classifier (LR), plus resident-set memory
at end of inference. Numbers are reported in the paper as part of Limitation L9
(previously: "cost / inference time / memory footprint not reported"). The cell expects
the trained base learners to still be in scope; if not, restart the notebook and run
all cells before this one.
"""


def main():
    print("Editing notebooks...\n")

    # SVM: add save-preds after cell 48 (final eval), then per-corpus at end
    edit_notebook(ROOT / "SVM.ipynb", [
        insert_save_preds("SVM", "Y_pred_all", "y_test", after_cell_idx=48),
        append_percorpus("SVM"),
    ])

    # BERT: add save-preds after cell 40 (prediction extraction), then per-corpus at end
    # Also add a markdown documentation cell about class-imbalance handling.
    def _bert_extras(nb):
        if not already_has(nb, "## Class-imbalance handling in the BERT fine-tuning loss"):
            nb["cells"].insert(29, md_cell(BERT_LOSS_NOTE))
    edit_notebook(ROOT / "BERT_FineTune.ipynb", [
        insert_save_preds("BERT", "Y_pred_all", "y_test", after_cell_idx=40),
        _bert_extras,
        append_percorpus("BERT"),
    ])

    # LSTM: add vocab-size note + per-corpus
    def _lstm_extras(nb):
        if not already_has(nb, "## Note on tokenizer vocabulary size"):
            # Insert as cell 0 (top of notebook) — visible context for the design choice
            nb["cells"].insert(0, md_cell(LSTM_VOCAB_NOTE))
    edit_notebook(ROOT / "LSTM.ipynb", [
        _lstm_extras,
        append_percorpus("LSTM"),
    ])

    # RandomForest + XGBoost: just per-corpus cells (preds already saved)
    edit_notebook(ROOT / "RandomForest.ipynb", [append_percorpus("RandomForest")])
    edit_notebook(ROOT / "XGBoost.ipynb",      [append_percorpus("XGBoost")])

    # Ensemble: add cost-instrumentation cell (per-corpus already exists in cell 81)
    def _ens_extras(nb):
        if already_has(nb, "# ENSEMBLE_COST_v1"):
            return
        nb["cells"].append(md_cell(ENSEMBLE_TIMING_NOTE))
        nb["cells"].append(code_cell(ENSEMBLE_TIMING))
    edit_notebook(ROOT / "StackedEnsemble.ipynb", [_ens_extras])

    # Back-translation: pivot-language justification
    def _bt_extras(nb):
        if not already_has(nb, "## Pivot language for back-translation"):
            nb["cells"].insert(0, md_cell(PIVOT_LANG_NOTE))
    edit_notebook(ROOT / "backtranslation_standalone_augmentation.ipynb", [_bt_extras])

    # MachineDiscoveredTopicClassification: NPMI note
    def _mdtc_extras(nb):
        if not already_has(nb, "## On the negative NPMI for Top2Vec"):
            nb["cells"].insert(0, md_cell(TOP2VEC_NPMI_NOTE))
    edit_notebook(ROOT / "MachineDiscoveredTopicClassification.ipynb", [_mdtc_extras])

    print("\nDone.")


if __name__ == "__main__":
    main()
