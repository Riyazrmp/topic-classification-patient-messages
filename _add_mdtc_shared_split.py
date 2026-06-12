"""Add a 'load shared split training corpus' cell to MachineDiscoveredTopicClassification.ipynb
so BERTopic / Top2Vec / LDA are all fit on the same dedup'd, filtered training corpus
used by the supervised models. Idempotent."""

import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "MachineDiscoveredTopicClassification.ipynb"
MARKER = "# SHARED_SPLIT_TOPIC_INPUT_v1"

LOADER_NOTE = """## Input corpus: shared training split

To keep the unsupervised topic models (BERTopic, Top2Vec, LDA) directly comparable
with the supervised pipeline, we fit them on the **training portion of the shared
split** produced by `create_shared_split.py` (dedup'd, restricted to labels with
>= 100 occurrences in the pooled corpus). The next cell reconstructs that input.
Replace the loose `df_diabetes` / `df_covid` reads further down with `train_docs`.
"""

LOADER_CODE = '''# SHARED_SPLIT_TOPIC_INPUT_v1
# Build the training-split corpus that the supervised models see, for fair comparison.
import random, numpy as np, pandas as pd

SEED = 42
random.seed(SEED); np.random.seed(SEED)

dfCovid    = pd.read_csv("data/dataRwaCovid.csv", on_bad_lines="skip", low_memory=False)
dfDiabetes = pd.read_csv("data/dataIHADiabetes.csv", on_bad_lines="skip", low_memory=False)

DiabetesTopics = ["Physical","Psychological","No Symptoms","Prognosis","Laboratory/Testing","Imaging","Clinical","Testing/Monitoring Devices","Health Data","Diagnostic Methods - Other","Medications","Procedures","Alternative","Physical Therapy","Counseling","Adverse Events","Therapeutic Devices","Treatment(Rx) - Other","Outpatient Logistics/Scheduling","Hospitalizations","Insurance/Billing","Medical Records","Referrals","Transportation","Primary (Pharmaceutical Prevention)","Primary (Non-Pharmaceutical Prevention)","Secondary (Pharmaceutical Prevention)","Secondary (Non-Pharmaceutical Prevention)","Diet/Nutrition","Exercise","Substance Use","Entertainment","Lifestyle - Other","Housing","Work/School","Social Services","Friends & Family","Cultural/Religion","Travel","Physical Environment/Climate","Financial","Social - Other","Technical/IT","Safety Concerns","Health Education","Sexual & Reproductive Health","Child & Family Health","Problems Solved","Grateful Patient","Service Complaint","Request to Stop","Emergent","Urgent","Non-urgent","Stigma Present","Rapport","Transition to Adult Clinic"]
CovidTopics = ["Physical","Mental/Emotional","No Symptoms","Laboratory/Testing","Imaging","Clinical","Diagnostic Methods - Other","Medications","Procedures","Alternative","Physical Therapy","Counseling","Treatment(Rx) - Other","Outpatient Logistics/Scheduling","Hospitalizations","Pharmaceutical Prevention","Non-Pharmaceutical Prevention","Diet/Nutrition","Exercise","Substance Use","Lifestyle - Other","Housing","Work/School","Social Services","Friends & Family","Cultural/Religion","Travel","Physical Environment/Climate","Financial","Social - Other","Technical/IT","Safety concern","Health Education","Maternal & Child Health","Problems Solved","Grateful Patient","Service Complaint","Request to Stop","Emergent","Urgent","Non-urgent","Stigma Present","wave","batch"]
common = sorted(set(DiabetesTopics) & set(CovidTopics))

dfD = dfDiabetes[["conversation"] + common].copy(); dfD["source"] = "Canada"
dfC = dfCovid[["conversation(english_only)"] + common].rename(
    columns={"conversation(english_only)": "conversation"}).copy(); dfC["source"] = "Rwanda"

dfCombined = pd.concat([dfC, dfD]).reset_index(drop=True)
label_keep = dfCombined[common].sum()[lambda s: s >= 100].index.tolist()
dfFilt = (dfCombined[["conversation", "source"] + label_keep]
          .drop_duplicates(subset="conversation")
          .reset_index(drop=True))

_split = np.load("data/shared_split_indices.npz")
train_idx = _split["train_idx"]

train_docs    = dfFilt.iloc[train_idx]["conversation"].tolist()
train_sources = dfFilt.iloc[train_idx]["source"].tolist()
print(f"Training docs for unsupervised models: {len(train_docs)} "
      f"(Rwanda {sum(s=='Rwanda' for s in train_sources)}, "
      f"Canada {sum(s=='Canada' for s in train_sources)})")
print(f"Labels kept (only used for downstream comparison): {len(label_keep)}")
'''


def _detect_indent(text: str) -> int:
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        if stripped.startswith('"cells"'):
            n = len(line) - len(stripped)
            return n if n in (1, 2, 3, 4) else 2
    return 2


def main():
    original_text = NB_PATH.read_text()
    indent = _detect_indent(original_text)
    nb = json.loads(original_text)
    if any(MARKER in ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
           for c in nb["cells"]):
        print(f"{NB_PATH.name}: already has shared-split loader, no change")
        return

    note = {"cell_type": "markdown", "metadata": {}, "source": LOADER_NOTE.splitlines(keepends=True)}
    code = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": LOADER_CODE.splitlines(keepends=True)}

    # Insert AFTER the NPMI markdown (cell 0), so notebook order is: NPMI note,
    # shared-split note, shared-split code, then the original cells.
    nb["cells"].insert(1, note)
    nb["cells"].insert(2, code)

    NB_PATH.write_text(json.dumps(nb, indent=indent, ensure_ascii=False))
    print(f"{NB_PATH.name}: inserted shared-split loader at cells 1-2; total cells now {len(nb['cells'])}")


if __name__ == "__main__":
    main()
