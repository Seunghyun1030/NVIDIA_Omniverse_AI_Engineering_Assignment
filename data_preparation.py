"""
Task 2 — Data Preparation
==========================
This script handles the complete data preparation pipeline:
1. Load diabetic_data.csv and IDS_mapping.csv
2. Apply ID mappings to decode categorical columns
3. Remove deceased/hospice patients (cannot be readmitted)
4. Perform initial data cleaning (missing values, duplicates, data types)
5. Feature engineering (ICD-9 grouping, medication aggregation, visit totals)
6. Drop low-information columns
7. Create a 5% stratified unseen sample (unseen_data.csv)
8. Split remaining 95% into 80% train / 20% test (stratified)
9. Save all splits to CSV files

References
----------
- Strack B, DeShazo JP, et al. "Impact of HbA1c Measurement on Hospital
  Readmission Rates." BioMed Research International, 2014.
- García-Mosquera J, et al. "Transformer-Based Prediction of Hospital
  Readmissions for Diabetes Patients." Electronics, 2025.
- Emi-Johnson OG, et al. "Predicting 30-Day Hospital Readmission in Patients
  With Diabetes Using ML on EHR Data." Cureus, 2025.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# ICD-9 Diagnosis Code Grouping (Strack et al., 2014)
# ============================================================
def map_diagnosis_to_category(diag_code: str) -> str:
    """
    Map an ICD-9 diagnosis code to a broad clinical category.

    Based on: Strack B, DeShazo JP, et al. "Impact of HbA1c Measurement
    on Hospital Readmission Rates." BioMed Research International, 2014.

    Parameters
    ----------
    diag_code : str
        Raw ICD-9 code (e.g. "250.83", "V58", "E819", "?").

    Returns
    -------
    str
        Clinical category: 'Diabetes', 'Circulatory', 'Respiratory',
        'Digestive', 'Injury', 'Musculoskeletal', 'Genitourinary',
        'Neoplasms', or 'Other'.
    """
    if diag_code is None or str(diag_code).strip() in ("", "?", "nan"):
        return "Other"

    diag_code = str(diag_code).strip()

    # V-codes and E-codes → Other
    if diag_code.startswith("V") or diag_code.startswith("E"):
        return "Other"

    try:
        code_num = float(diag_code)
    except ValueError:
        return "Other"

    if 250 <= code_num < 251:
        return "Diabetes"
    if (390 <= code_num <= 459) or (785 <= code_num < 786):
        return "Circulatory"
    if (460 <= code_num <= 519) or (786 <= code_num < 787):
        return "Respiratory"
    if (520 <= code_num <= 579) or (787 <= code_num < 788):
        return "Digestive"
    if (580 <= code_num <= 629) or (788 <= code_num < 789):
        return "Genitourinary"
    if 800 <= code_num <= 999:
        return "Injury"
    if 710 <= code_num <= 739:
        return "Musculoskeletal"
    if 140 <= code_num <= 239:
        return "Neoplasms"

    return "Other"


def convert_age_to_numeric(age_str: str) -> int:
    """
    Convert age range string to numeric midpoint.

    Parameters
    ----------
    age_str : str
        Age range like "[70-80)".

    Returns
    -------
    int
        Midpoint of the range (e.g. 75).
    """
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    return age_map.get(str(age_str), 65)  # default to median age


def group_medical_specialty(specialty: str) -> str:
    """
    Group 73 medical specialties into broader clinical categories.

    Parameters
    ----------
    specialty : str
        Raw medical_specialty value.

    Returns
    -------
    str
        Grouped category.
    """
    if specialty is None or str(specialty).strip() in ("", "nan"):
        return "Unknown"

    s = str(specialty).strip()

    top_specialties = {
        "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice",
        "Cardiology", "Surgery-General", "Nephrology", "Orthopedics",
        "Orthopedics-Reconstructive", "Radiologist", "Pulmonology",
    }

    if s in top_specialties:
        return s
    return "Other"


# ============================================================
# Step 1: Load datasets
# ============================================================
DATA_DIR = "data_diabetes_hospital_readmission_1999-2008"

df = pd.read_csv(f"{DATA_DIR}/diabetic_data.csv")
ids_raw = pd.read_csv(f"{DATA_DIR}/IDS_mapping.csv")

print(f"Original dataset shape: {df.shape}")
print(f"Target distribution:\n{df['readmitted'].value_counts()}\n")

# ============================================================
# Step 2: Parse IDS_mapping.csv into separate lookup tables
# ============================================================
# The IDS_mapping.csv contains 3 tables stacked vertically,
# separated by rows where admission_type_id contains the next table's header.

ids_raw.columns = ["id", "description"]

# Find separator rows (where 'id' column contains a string header)
sep_indices = ids_raw[ids_raw["id"].isin([
    "discharge_disposition_id", "admission_source_id"
])].index.tolist()

# Split into 3 mapping tables
admission_type_map = ids_raw.iloc[:sep_indices[0]].dropna().copy()
admission_type_map["id"] = admission_type_map["id"].astype(int)

discharge_map = ids_raw.iloc[sep_indices[0]+1:sep_indices[1]].dropna().copy()
discharge_map["id"] = discharge_map["id"].astype(int)

admission_source_map = ids_raw.iloc[sep_indices[1]+1:].dropna().copy()
admission_source_map["id"] = admission_source_map["id"].astype(int)

# Apply mappings to decode categorical IDs
admission_type_dict = dict(zip(admission_type_map["id"], admission_type_map["description"]))
discharge_dict = dict(zip(discharge_map["id"], discharge_map["description"]))
admission_source_dict = dict(zip(admission_source_map["id"], admission_source_map["description"]))

df["admission_type_id"] = df["admission_type_id"].map(admission_type_dict).fillna("Unknown")
df["discharge_disposition_id"] = df["discharge_disposition_id"].map(discharge_dict).fillna("Unknown")
df["admission_source_id"] = df["admission_source_id"].map(admission_source_dict).fillna("Unknown")

print("ID mappings applied successfully.")
print(f"  admission_type_id unique values: {df['admission_type_id'].nunique()}")
print(f"  discharge_disposition_id unique values: {df['discharge_disposition_id'].nunique()}")
print(f"  admission_source_id unique values: {df['admission_source_id'].nunique()}\n")

# ============================================================
# Step 2.5: Remove deceased / hospice patients
# ============================================================
# Patients who expired or were discharged to hospice cannot be readmitted.
# This is standard practice in the literature:
#   - Strack et al. (2014): removed expired patients
#   - García-Mosquera et al. (2025): "Patients discharged due to death or
#     palliative care were likewise removed, as they could not be readmitted."
#   - Emi-Johnson et al. (2025): "Patients who were discharged to hospice or
#     who expired during hospitalization were excluded."
#
# Original discharge_disposition_id values to remove:
#   11 = Expired, 13 = Hospice / home, 14 = Hospice / medical facility,
#   19 = Expired at home, 20 = Expired in a medical facility,
#   21 = Expired / place unknown

expired_keywords = ["Expired", "Hospice"]
expired_mask = df["discharge_disposition_id"].str.contains(
    "|".join(expired_keywords), case=False, na=False
)
n_expired = expired_mask.sum()
df = df[~expired_mask].copy()

print(f"Removed {n_expired} deceased/hospice patients (cannot be readmitted).")
print(f"  Dataset shape after removal: {df.shape}")
print(f"  Target distribution after removal:\n{df['readmitted'].value_counts()}\n")

# ============================================================
# Step 3: Data Cleaning
# ============================================================

# 3a. Replace '?' with NaN for proper missing value handling
df.replace("?", np.nan, inplace=True)

# Count missing values per column
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_cols = missing_report[missing_report["missing_count"] > 0]
print("Columns with missing values (after replacing '?'):")
print(missing_cols.to_string())
print()

# 3b. Remove duplicate rows based on encounter_id (each encounter should be unique)
n_before = len(df)
df.drop_duplicates(subset=["encounter_id"], keep="first", inplace=True)
n_after = len(df)
print(f"Duplicates removed (by encounter_id): {n_before - n_after}")

# 3c. Remove invalid gender rows
n_before = len(df)
df = df[df["gender"] != "Unknown/Invalid"].copy()
n_after = len(df)
print(f"Removed {n_before - n_after} rows with invalid gender")

# 3d. Create patient encounter count BEFORE dropping IDs
# Patients with more encounters are far more likely to be readmitted
# (García-Mosquera et al. 2025: number_inpatient was the #1 predictor)
#
# NOTE on patient duplication:
# The same patient (patient_nbr) may appear multiple times in this dataset.
# Ideally, train/test splits should be done at the patient level to prevent
# data leakage (as in Liu et al. 2024 with group k-fold CV). However, the
# assignment specifies a standard stratified split on 'readmitted', so we
# follow that instruction. We create num_prior_encounters to capture this
# information as a feature instead.
df["num_prior_encounters"] = df.groupby("patient_nbr")["encounter_id"].transform("count")
print(f"Created num_prior_encounters (max: {df['num_prior_encounters'].max()}, "
      f"mean: {df['num_prior_encounters'].mean():.2f})")

# 3e. Drop columns that are identifiers (not useful for prediction)
df.drop(columns=["encounter_id", "patient_nbr"], inplace=True)
print(f"Dropped identifier columns: encounter_id, patient_nbr")
print(f"Cleaned dataset shape: {df.shape}\n")

# ============================================================
# Step 3.5: Feature Engineering
# ============================================================

# 3.5a. Convert age ranges to numeric midpoints
df["age"] = df["age"].apply(convert_age_to_numeric)
print(f"Age converted to numeric (range: {df['age'].min()}-{df['age'].max()})")

# 3.5b. Group medical_specialty (73 unique → top 10 + Other + Unknown)
df["medical_specialty"] = df["medical_specialty"].apply(group_medical_specialty)
print(f"Medical specialty grouped ({df['medical_specialty'].nunique()} categories)")

# 3.5c. ICD-9 diagnosis code grouping (Strack et al., 2014)
# Reduce 700+ unique codes to 9 clinical categories
for col in ["diag_1", "diag_2", "diag_3"]:
    df[col + "_category"] = df[col].apply(map_diagnosis_to_category)

# Drop original high-cardinality diagnosis columns
df.drop(columns=["diag_1", "diag_2", "diag_3"], inplace=True)
print("ICD-9 diagnosis codes grouped into clinical categories.")

# 3.5d. Keep A1Cresult and max_glu_serum as categorical features
# These are clinically important despite high missingness (~47% and ~95%).
# Strack et al. (2014) specifically studied HbA1c's impact on readmission.
# García-Mosquera et al. (2025) found A1c test result more relevant than
# glucose serum test for tracking patient outcomes.
#
# The test-vs-no-test distinction itself carries predictive signal:
# patients who received HbA1c testing may have different care patterns.
df["A1Cresult"] = df["A1Cresult"].fillna("Not Tested")
df["max_glu_serum"] = df["max_glu_serum"].fillna("Not Tested")
print(f"A1Cresult kept as categorical: {df['A1Cresult'].value_counts().to_dict()}")
print(f"max_glu_serum kept as categorical: {df['max_glu_serum'].value_counts().to_dict()}")

# 3.5e. Medication aggregation — 23 drug columns → summary features
medication_cols = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Count how many medications the patient is on (not "No" and not "Steady")
df["num_active_meds"] = df[medication_cols].apply(
    lambda row: sum(1 for v in row if str(v) not in ("No", "nan", "Steady")), axis=1
)

# Count how many medications were changed (Up or Down)
df["num_med_changes"] = df[medication_cols].apply(
    lambda row: sum(1 for v in row if str(v) in ("Up", "Down")), axis=1
)

# Count how many medications are steady
df["num_steady_meds"] = df[medication_cols].apply(
    lambda row: sum(1 for v in row if str(v) == "Steady"), axis=1
)

# Drop individual medication columns (keep insulin as it's the most important)
# García-Mosquera et al. (2025): insulin had the highest importance weight
# among all medication features.
med_cols_to_drop = [c for c in medication_cols if c != "insulin"]
df.drop(columns=med_cols_to_drop, inplace=True)
print(f"Medication columns aggregated. Dropped {len(med_cols_to_drop)} individual drug columns, kept 'insulin'.")

# 3.5f. Visit history aggregation
df["total_visits"] = (
    df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
)
print("Total visits feature created.")

# 3.5g. Interaction features
df["med_per_day"] = df["num_medications"] / df["time_in_hospital"].clip(lower=1)
df["inpatient_plus_time"] = df["number_inpatient"] + df["time_in_hospital"]
print("Interaction features created: med_per_day, inpatient_plus_time")

# 3.5h. Drop columns with extreme missingness or no predictive value
# - weight: ~97% missing — nearly all values are absent
# - payer_code: ~40% missing — insurance type has limited clinical relevance
#   for readmission prediction; also not available in many clinical settings
# NOTE: A1Cresult and max_glu_serum are intentionally KEPT (see 3.5d above)
high_missing_cols = ["weight", "payer_code"]
df.drop(columns=high_missing_cols, inplace=True)
print(f"Dropped high-missingness columns: {high_missing_cols}")

print(f"\nFinal dataset shape after feature engineering: {df.shape}")
print(f"Final columns ({len(df.columns)}):")
print(f"  {list(df.columns)}\n")

# ============================================================
# Step 4: Stratified 5% unseen sample
# ============================================================
df_working, df_unseen = train_test_split(
    df,
    test_size=0.05,
    random_state=42,
    stratify=df["readmitted"]
)

df_unseen.to_csv("unseen_data.csv", index=False)
print(f"Unseen sample shape: {df_unseen.shape}")
print(f"Unseen target distribution:\n{df_unseen['readmitted'].value_counts()}\n")

# ============================================================
# Step 5: 80/20 stratified train/test split on remaining 95%
# ============================================================
df_train, df_test = train_test_split(
    df_working,
    test_size=0.20,
    random_state=42,
    stratify=df_working["readmitted"]
)

df_train.to_csv("train_data.csv", index=False)
df_test.to_csv("test_data.csv", index=False)

print(f"Training set shape: {df_train.shape}")
print(f"Training target distribution:\n{df_train['readmitted'].value_counts()}\n")

print(f"Test set shape: {df_test.shape}")
print(f"Test target distribution:\n{df_test['readmitted'].value_counts()}\n")

# ============================================================
# Summary
# ============================================================
print("=" * 50)
print("DATA SPLIT SUMMARY")
print("=" * 50)
print(f"Full dataset:    {len(df):>7} rows")
print(f"Unseen (5%):     {len(df_unseen):>7} rows  -> unseen_data.csv")
print(f"Working (95%):   {len(df_working):>7} rows")
print(f"  Train (80%):   {len(df_train):>7} rows  -> train_data.csv")
print(f"  Test  (20%):   {len(df_test):>7} rows  -> test_data.csv")
print("=" * 50)
