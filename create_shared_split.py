

import random
import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

dfCovid    = pd.read_csv("data/dataRwaCovid.csv")
dfDiabetes = pd.read_csv("data/dataIHADiabetes.csv")

DiabetesTopics = [
    "Physical", "Psychological", "No Symptoms", "Prognosis",
    "Laboratory/Testing", "Imaging", "Clinical", "Testing/Monitoring Devices",
    "Health Data", "Diagnostic Methods - Other", "Medications", "Procedures",
    "Alternative", "Physical Therapy", "Counseling", "Adverse Events",
    "Therapeutic Devices", "Treatment(Rx) - Other", "Outpatient Logistics/Scheduling",
    "Hospitalizations", "Insurance/Billing", "Medical Records", "Referrals",
    "Transportation", "Primary (Pharmaceutical Prevention)",
    "Primary (Non-Pharmaceutical Prevention)", "Secondary (Pharmaceutical Prevention)",
    "Secondary (Non-Pharmaceutical Prevention)", "Diet/Nutrition", "Exercise",
    "Substance Use", "Entertainment", "Lifestyle - Other", "Housing", "Work/School",
    "Social Services", "Friends & Family", "Cultural/Religion", "Travel",
    "Physical Environment/Climate", "Financial", "Social - Other", "Technical/IT",
    "Safety Concerns", "Health Education", "Sexual & Reproductive Health",
    "Child & Family Health", "Problems Solved", "Grateful Patient",
    "Service Complaint", "Request to Stop", "Emergent", "Urgent", "Non-urgent",
    "Stigma Present", "Rapport", "Transition to Adult Clinic",
]

CovidTopics = [
    "Physical", "Mental/Emotional", "No Symptoms", "Laboratory/Testing", "Imaging",
    "Clinical", "Diagnostic Methods - Other", "Medications", "Procedures",
    "Alternative", "Physical Therapy", "Counseling", "Treatment(Rx) - Other",
    "Outpatient Logistics/Scheduling", "Hospitalizations",
    "Pharmaceutical Prevention", "Non-Pharmaceutical Prevention", "Diet/Nutrition",
    "Exercise", "Substance Use", "Lifestyle - Other", "Housing", "Work/School",
    "Social Services", "Friends & Family", "Cultural/Religion", "Travel",
    "Physical Environment/Climate", "Financial", "Social - Other", "Technical/IT",
    "Safety concern", "Health Education", "Maternal & Child Health",
    "Problems Solved", "Grateful Patient", "Service Complaint", "Request to Stop",
    "Emergent", "Urgent", "Non-urgent", "Stigma Present", "wave", "batch",
]

commonSubtopics = sorted(set(DiabetesTopics) & set(CovidTopics))

diabetesColumns = ["conversation"] + commonSubtopics
covidColumns    = ["conversation(english_only)"] + commonSubtopics

dfDiabetesNew = dfDiabetes[diabetesColumns]
dfCovidNew    = dfCovid[covidColumns].rename(columns={"conversation(english_only)": "conversation"})

dfCovidNew    = dfCovidNew.copy();    dfCovidNew["source"]    = "Rwanda"
dfDiabetesNew = dfDiabetesNew.copy(); dfDiabetesNew["source"] = "Canada"

dfCombined = pd.concat([dfCovidNew, dfDiabetesNew]).reset_index(drop=True)

n = 100
label_sum            = dfCombined[commonSubtopics].sum()
label_keep_original  = label_sum[label_sum >= n].index.tolist()

dfCombined_filtered = dfCombined[["conversation", "source"] + label_keep_original].copy()
dfCombined_filtered = (
    dfCombined_filtered
    .drop_duplicates(subset='conversation')
    .reset_index(drop=True)
)
print(f"After dedup: {len(dfCombined_filtered)}")
# dfCombined_filtered = dfCombined_filtered.reset_index(drop=True)   # <- canonical index

print(f"dfCombined_filtered shape: {dfCombined_filtered.shape}")
print(f"Labels kept ({len(label_keep_original)}): {label_keep_original}")

N   = len(dfCombined_filtered)
idx = np.arange(N).reshape(-1, 1)          # integer positions — what we'll split
y   = dfCombined_filtered[label_keep_original].to_numpy()

np.random.seed(SEED)
train_idx, y_train, holdout_idx, y_holdout = iterative_train_test_split(
    idx, y, test_size=0.30
)
train_idx   = train_idx.flatten()
holdout_idx = holdout_idx.flatten()

val_idx, _, test_idx, _ = iterative_train_test_split(
    holdout_idx.reshape(-1, 1), y_holdout, test_size=0.50
)
val_idx  = val_idx.flatten()
test_idx = test_idx.flatten()

print(f"\nSplit sizes:")
print(f"  train : {len(train_idx):>5}  ({100*len(train_idx)/N:.1f} %)")
print(f"  val   : {len(val_idx):>5}  ({100*len(val_idx)/N:.1f} %)")
print(f"  test  : {len(test_idx):>5}  ({100*len(test_idx)/N:.1f} %)")

np.savez(
    "data/shared_split_indices.npz",
    train_idx = train_idx,
    val_idx   = val_idx,
    test_idx  = test_idx,
)
print("\nSaved → data/shared_split_indices.npz")
