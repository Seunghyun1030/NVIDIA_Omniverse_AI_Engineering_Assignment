"""
Inference Script — Grader Usage
================================
Run predictions on unseen_data.csv using the trained AutoGluon model.

This script automatically detects whether the input data is raw (original format)
or already preprocessed, and applies the necessary feature engineering pipeline.

Usage:
    python predict.py                          # default: unseen_data.csv -> predictions.csv
    python predict.py --input my_data.csv      # custom input file
    python predict.py --output results.csv     # custom output file

Requirements:
    - Trained model in ag_models/ directory
    - Python 3.12 with packages from requirements.txt
"""

import argparse
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

TARGET = "readmitted"
IDS_MAPPING_PATH = "data_diabetes_hospital_readmission_1999-2008/IDS_mapping.csv"


# ============================================================
# Feature Engineering Pipeline (mirrors data_preparation.py)
# ============================================================
def map_diagnosis_to_category(diag_code: str) -> str:
    """Map ICD-9 code to clinical category (Strack et al., 2014)."""
    if diag_code is None or str(diag_code).strip() in ("", "?", "nan"):
        return "Other"
    diag_code = str(diag_code).strip()
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
    """Convert age range string to numeric midpoint."""
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    return age_map.get(str(age_str), 65)


def group_medical_specialty(specialty: str) -> str:
    """Group 73 specialties into top 10 + Other + Unknown."""
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


def is_raw_data(df: pd.DataFrame) -> bool:
    """Detect if data is in raw original format (needs preprocessing)."""
    raw_indicators = ["encounter_id", "patient_nbr", "diag_1", "diag_2", "diag_3"]
    return any(col in df.columns for col in raw_indicators)


def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to raw data.
    Mirrors the steps in data_preparation.py so that graders can
    feed in original-format CSV and get correct predictions.
    """
    print("  Raw data detected — applying feature engineering pipeline...")
    import os

    # Step 1: Apply ID mappings if columns are numeric
    if df["admission_type_id"].dtype in ("int64", "float64"):
        if os.path.exists(IDS_MAPPING_PATH):
            ids_raw = pd.read_csv(IDS_MAPPING_PATH)
            ids_raw.columns = ["id", "description"]
            sep_indices = ids_raw[ids_raw["id"].isin([
                "discharge_disposition_id", "admission_source_id"
            ])].index.tolist()

            admission_type_map = ids_raw.iloc[:sep_indices[0]].dropna().copy()
            admission_type_map["id"] = admission_type_map["id"].astype(int)
            discharge_map = ids_raw.iloc[sep_indices[0]+1:sep_indices[1]].dropna().copy()
            discharge_map["id"] = discharge_map["id"].astype(int)
            admission_source_map = ids_raw.iloc[sep_indices[1]+1:].dropna().copy()
            admission_source_map["id"] = admission_source_map["id"].astype(int)

            df["admission_type_id"] = df["admission_type_id"].map(
                dict(zip(admission_type_map["id"], admission_type_map["description"]))
            ).fillna("Unknown")
            df["discharge_disposition_id"] = df["discharge_disposition_id"].map(
                dict(zip(discharge_map["id"], discharge_map["description"]))
            ).fillna("Unknown")
            df["admission_source_id"] = df["admission_source_id"].map(
                dict(zip(admission_source_map["id"], admission_source_map["description"]))
            ).fillna("Unknown")
            print("    ID mappings applied.")

    # Step 2: Remove deceased/hospice patients
    expired_keywords = ["Expired", "Hospice"]
    expired_mask = df["discharge_disposition_id"].str.contains(
        "|".join(expired_keywords), case=False, na=False
    )
    n_expired = expired_mask.sum()
    if n_expired > 0:
        df = df[~expired_mask].copy()
        print(f"    Removed {n_expired} deceased/hospice patients.")

    # Step 3: Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Step 4: Remove invalid gender
    df = df[df["gender"] != "Unknown/Invalid"].copy()

    # Step 5: Create num_prior_encounters if patient_nbr exists
    if "patient_nbr" in df.columns and "encounter_id" in df.columns:
        df["num_prior_encounters"] = df.groupby("patient_nbr")["encounter_id"].transform("count")
        df.drop(columns=["encounter_id", "patient_nbr"], inplace=True)
    elif "patient_nbr" in df.columns:
        df["num_prior_encounters"] = 1  # single encounter, no history
        df.drop(columns=["patient_nbr"], inplace=True)
    elif "encounter_id" in df.columns:
        df.drop(columns=["encounter_id"], inplace=True)

    if "num_prior_encounters" not in df.columns:
        df["num_prior_encounters"] = 1

    # Step 6: Age to numeric
    if df["age"].dtype == "object":
        df["age"] = df["age"].apply(convert_age_to_numeric)

    # Step 7: Group medical_specialty
    df["medical_specialty"] = df["medical_specialty"].apply(group_medical_specialty)

    # Step 8: ICD-9 grouping
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col + "_category"] = df[col].apply(map_diagnosis_to_category)
            df.drop(columns=[col], inplace=True)

    # Step 9: A1Cresult and max_glu_serum — keep as categorical
    if "A1Cresult" in df.columns:
        df["A1Cresult"] = df["A1Cresult"].fillna("Not Tested")
    if "max_glu_serum" in df.columns:
        df["max_glu_serum"] = df["max_glu_serum"].fillna("Not Tested")

    # Step 10: Medication aggregation
    medication_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "examide",
        "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    existing_med_cols = [c for c in medication_cols if c in df.columns]
    if len(existing_med_cols) > 5:  # raw data has all medication columns
        df["num_active_meds"] = df[existing_med_cols].apply(
            lambda row: sum(1 for v in row if str(v) not in ("No", "nan", "Steady")), axis=1
        )
        df["num_med_changes"] = df[existing_med_cols].apply(
            lambda row: sum(1 for v in row if str(v) in ("Up", "Down")), axis=1
        )
        df["num_steady_meds"] = df[existing_med_cols].apply(
            lambda row: sum(1 for v in row if str(v) == "Steady"), axis=1
        )
        med_cols_to_drop = [c for c in existing_med_cols if c != "insulin"]
        df.drop(columns=med_cols_to_drop, inplace=True)

    # Step 11: Visit history aggregation
    if "total_visits" not in df.columns:
        if all(c in df.columns for c in ["number_outpatient", "number_emergency", "number_inpatient"]):
            df["total_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]

    # Step 12: Interaction features
    if "med_per_day" not in df.columns:
        if all(c in df.columns for c in ["num_medications", "time_in_hospital"]):
            df["med_per_day"] = df["num_medications"] / df["time_in_hospital"].clip(lower=1)
    if "inpatient_plus_time" not in df.columns:
        if all(c in df.columns for c in ["number_inpatient", "time_in_hospital"]):
            df["inpatient_plus_time"] = df["number_inpatient"] + df["time_in_hospital"]

    # Step 13: Drop high-missingness columns
    for col in ["weight", "payer_code"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    print(f"    Feature engineering complete. Shape: {df.shape}")
    return df


# ============================================================
# Parse command-line arguments
# ============================================================
parser = argparse.ArgumentParser(description="Run predictions on unseen patient data.")
parser.add_argument(
    "--input",
    type=str,
    default="unseen_data.csv",
    help="Path to input CSV file (default: unseen_data.csv)",
)
parser.add_argument(
    "--output",
    type=str,
    default="predictions.csv",
    help="Path to output CSV file (default: predictions.csv)",
)
parser.add_argument(
    "--model-path",
    type=str,
    default="ag_models/",
    help="Path to AutoGluon model directory (default: ag_models/)",
)
args = parser.parse_args()

# ============================================================
# Load model and data
# ============================================================
print(f"Loading model from {args.model_path}...")
predictor = TabularPredictor.load(args.model_path)

print(f"Reading input data from {args.input}...")
data = pd.read_csv(args.input)
print(f"  Input shape: {data.shape[0]} rows × {data.shape[1]} cols")

# ============================================================
# Auto-detect and preprocess if raw data
# ============================================================
if is_raw_data(data):
    data = preprocess_raw_data(data)

# Remove target column if present
has_target = TARGET in data.columns
if has_target:
    X = data.drop(columns=[TARGET])
else:
    X = data

# Remove sample_weight column if present (from training data)
if "sample_weight" in X.columns:
    X = X.drop(columns=["sample_weight"])

# ============================================================
# Run predictions (use fast L1 model for speed — <2 min vs 30+ min for ensemble)
# ============================================================
# LightGBMLarge_BAG_L1 is the top single model (F1 0.576 vs ensemble 0.586)
# Using it directly avoids the slow multi-level ensemble prediction chain
FAST_MODEL = "LightGBMLarge_BAG_L1"
available_models = predictor.model_names()
use_model = FAST_MODEL if FAST_MODEL in available_models else None

if use_model:
    print(f"Running predictions with {use_model} (fast single model)...")
else:
    print("Running predictions with best model...")

predictions = predictor.predict(X, model=use_model)

# Build output DataFrame
output_df = data.copy()
output_df["predicted_readmitted"] = predictions.values

# Add prediction probabilities
try:
    probabilities = predictor.predict_proba(X, model=use_model)
    for col in probabilities.columns:
        output_df[f"prob_{col}"] = probabilities[col].values
except Exception as e:
    print(f"  Note: predict_proba skipped ({e})")

output_df.to_csv(args.output, index=False)
print(f"\nPredictions saved to {args.output}")
print(f"  Output shape: {output_df.shape[0]} rows × {output_df.shape[1]} cols")
print(f"\nPrediction distribution:")
print(predictions.value_counts().to_string())

# If target column was present, show quick accuracy check
if has_target:
    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(data[TARGET], predictions)
    f1 = f1_score(data[TARGET], predictions, average="macro", zero_division=0)
    print(f"\nQuick evaluation (target column found):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Macro:  {f1:.4f}")
