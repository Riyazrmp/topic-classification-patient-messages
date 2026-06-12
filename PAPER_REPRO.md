# Paper reproduction handoff (May 2026 submission round)

Everything below is what the submitted paper needs. The codebase now has the
infrastructure for per-corpus + per-label F1 with bootstrap 95% CIs, plus the
documentation cells the reviewers will ask for. Some numbers still need a
re-run on **your** GPU machine before the manuscript is final.

---

## TL;DR for Riyaz

On your GPU box, run these three notebooks end-to-end, then re-run
`per_corpus_evaluation.py`:

1. `SVM.ipynb`             (CPU; saves `predictions/SVM_preds.npz`)
2. `BERT_FineTune.ipynb`   (GPU; saves `predictions/BERT_preds.npz`)
3. `MachineDiscoveredTopicClassification.ipynb`  (CPU or GPU; produces BERTopic / Top2Vec results on the shared training split)
4. `StackedEnsemble.ipynb` — only the last new cell (`# ENSEMBLE_COST_v1`) needs to be re-run on the same machine the trained base learners live on, to populate `results/ensemble_cost.json`.

Then:
```bash
python per_corpus_evaluation.py     # refreshes results/per_corpus_*.csv
python _paper_summary.py            # rebuilds results/paper_table_*.md
git add predictions/ results/ && git commit -m "rerun: SVM + BERT preds + BERTopic + ensemble cost"
```

The PR (`shera/per-corpus-eval`) already contains:

- per-corpus eval driver (`per_corpus_evaluation.py`)
- paper-table generator (`_paper_summary.py`)
- per-corpus cells appended to every model notebook
- save-preds cells inserted into SVM + BERT (so the rerun produces the missing `*_preds.npz`)
- markdown notes covering: LSTM vocab size, back-translation pivot language, BERT
  class-imbalance handling, Top2Vec negative-NPMI explanation
- shared-split loader for the unsupervised topic notebook
- inference-cost instrumentation cell at the end of `StackedEnsemble.ipynb`

---

## What's already in `results/` (done without rerun)

Built from the saved predictions for **LSTM, RandomForest, XGBoost, Ensemble**:

| Model        | Combined           | Rwanda             | Canada             |
|--------------|--------------------|--------------------|--------------------|
| Ensemble     | 0.78 [0.75, 0.81]  | 0.68 [0.64, 0.71]  | 0.63 [0.58, 0.67]  |
| RandomForest | 0.69 [0.66, 0.72]  | 0.55 [0.51, 0.58]  | 0.62 [0.55, 0.65]  |
| XGBoost      | 0.69 [0.66, 0.72]  | 0.56 [0.52, 0.60]  | 0.64 [0.56, 0.67]  |
| LSTM         | 0.58 [0.54, 0.61]  | 0.42 [0.36, 0.47]  | 0.50 [0.45, 0.53]  |

Test set: 584 docs (Rwanda 461, Canada 123).

The **gap between pooled and per-corpus** is the headline finding for the
revised Discussion. The Ensemble's "Macro F1 = 0.78" pooled drops to 0.68
on the Rwandan half and 0.63 on the Canadian half — the cross-context
generalization claim in the current draft must be softened accordingly.

Files:
- `results/per_corpus_macro_f1.csv`        Macro F1 + bootstrap CI per model x corpus
- `results/per_corpus_per_label_f1.csv`    F1 per label per corpus, with support counts
- `results/bootstrap_macro_f1.csv`         Raw 1,000-resample bootstrap distributions
- `results/paper_table_macro_by_corpus.md` Paste straight into the manuscript
- `results/paper_table_safety_critical.md` F1 on Urgent / Emergent / Service Complaint

---

## What still needs your machine

### 1. SVM per-corpus numbers
The SVM notebook trains via Optuna (50 trials x 19 labels) which is too slow
to rerun locally. The new save-preds cell now writes `predictions/SVM_preds.npz`
at the end of the notebook. Re-run `SVM.ipynb` once; the per-corpus eval picks
it up automatically.

### 2. BERT per-corpus numbers
Same deal: the predictions weren't saved. New cell after the prediction
extraction (cell 41 area) writes `predictions/BERT_preds.npz`. Plus a markdown
cell documenting the `BCEWithLogitsLoss(pos_weight=N_neg/N_pos)` recipe (which
is what we say in the Methods section when reviewers ask how class imbalance is
handled).

### 3. BERTopic / Top2Vec on the shared split
`MachineDiscoveredTopicClassification.ipynb` now starts with:
- a markdown cell explaining the negative NPMI for Top2Vec (Limitations L7-adjacent),
- a shared-split loader that produces `train_docs` from the same dedup'd,
  >=100-occurrence filtered corpus the supervised models see.

You'll need to wire the existing BERTopic / Top2Vec / LDA cells to consume
`train_docs` (currently they read raw `df_diabetes` / `df_covid` CSVs). Once
they're fit on `train_docs`, compute NPMI on the same corpus and report
side-by-side. If Top2Vec still produces negative NPMI, the explanation cell
covers it; if positive on the new split, revise the explanation.

### 4. Ensemble inference cost
`StackedEnsemble.ipynb` now ends with a cell (`# ENSEMBLE_COST_v1`) that times
each base learner + the meta-classifier on the held-out test set and reports
RSS memory. It needs the trained models still in scope — restart the notebook
and run all cells, then it writes `results/ensemble_cost.json`. (Variable names
may need a 30-second tweak depending on how SVM/XGBoost/BERT predictions are
re-derived in your current cells; the comment in the cell lists what it expects.)

---

## Manuscript edits that depend on the new numbers

When the above three reruns finish:

1. **Abstract / Results**: replace the old single Macro F1 with
   `0.78 [0.75, 0.81]` pooled and add a sentence about the per-corpus drop.
2. **Discussion** "cross-context generalization" paragraph: soften — pool-vs-corpus
   gap of ~10-15 F1 points means we are NOT claiming uniform cross-site
   performance. Recommend human-in-the-loop only.
3. **Per-label results**: Emergent improves to 0.47 (was 0.32 in the v5 draft),
   Urgent stays at 0.38, Service Complaint drops to 0.57 (was 0.67).
   The "central limitation" framing from the RevisionGuide still applies but
   with the corrected numbers.
4. **Limitations section**: L1 (single split / no CV) and L2 (no CIs) are
   partially addressed — now we DO report bootstrap CIs; the single-split caveat
   stands. L9 (cost not reported) becomes resolvable once the Ensemble cost cell
   runs.
5. **Table 1** support row: 584-doc test set, Rwanda 461, Canada 123.
6. **Safety-critical labels**: paste `results/paper_table_safety_critical.md`
   into the manuscript and discuss the Canada=0 cells (Emergent and
   Service Complaint each have <=1 positive in the Canadian test fold; that is
   not a model failure, it's a support issue).

---

## What's NOT in this PR

Manuscript-level edits (LaTeX). Those are tracked in `TopicPaper_RevisionGuide.pdf`
on Shera's side and will be applied to `TopicPaper_v4.tex` after this PR is
merged + the missing numbers are in.

---

## Re-running the per-corpus eval from scratch

```bash
# (one-time setup)
python -m venv .venv
.venv/bin/pip install numpy pandas scikit-learn scikit-multilearn

# whenever any *_preds.npz changes
.venv/bin/python per_corpus_evaluation.py
.venv/bin/python _paper_summary.py
```

The driver loads the Ensemble JSON for source labels (`source_corpus`), then
joins each model's saved `y_true / y_pred` matrix to that source vector for
the Rwanda / Canada splits. Row order is verified to match across all
prediction files.
