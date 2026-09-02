"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================
"""

import os
import pandas as pd
import mlflow

# === MODEL LOADING CONFIGURATION ===
# Model is copied into the Docker container at /app/model
MODEL_DIR = "/app/model"

try:
    # Load the trained XGBoost model in MLflow pyfunc format
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")

except Exception as e:
    raise Exception(
        f"❌ Failed to load model from {MODEL_DIR}: {e}"
    )

# === FEATURE SCHEMA LOADING ===
# feature_columns.txt is also copied into /app/model
try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")

    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]

    print(
        f"✅ Loaded {len(FEATURE_COLS)} feature columns from training"
    )

except Exception as e:
    raise Exception(
        f"❌ Failed to load feature columns: {e}"
    )


# === FEATURE TRANSFORMATION CONSTANTS ===

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    """

    df = df.copy()

    # === STEP 1: Clean column names ===
    df.columns = df.columns.str.strip()

    # === STEP 2: Numeric Type Coercion ===
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

            df[c] = df[c].fillna(0)

    # === STEP 3: Binary Feature Encoding ===
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    # === STEP 4: One-Hot Encoding ===
    obj_cols = [
        c
        for c in df.select_dtypes(
            include=["object"]
        ).columns
    ]

    if obj_cols:
        df = pd.get_dummies(
            df,
            columns=obj_cols,
            drop_first=True
        )

    # === STEP 5: Boolean to Integer ===
    bool_cols = df.select_dtypes(
        include=["bool"]
    ).columns

    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # === STEP 6: Feature Alignment ===
    df = df.reindex(
        columns=FEATURE_COLS,
        fill_value=0
    )

    return df


def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.
    """

    # === STEP 1: Convert input to DataFrame ===
    df = pd.DataFrame([input_dict])

    # === STEP 2: Transform features ===
    df_enc = _serve_transform(df)

    # === STEP 3: Generate prediction ===
    try:
        preds = model.predict(df_enc)

        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds

    except Exception as e:
        raise Exception(
            f"Model prediction failed: {e}"
        )

    # === STEP 4: Business-friendly output ===
    if result == 1:
        return "Likely to churn"
    else:
        return "Not likely to churn"