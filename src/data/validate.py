from pathlib import Path
import sys

import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("data/raw/transactions.csv")

REQUIRED_COLUMNS = [
    "amount",
    "account_age",
    "transaction_count",
    "fraud",
]

TARGET_COLUMN = "fraud"

MIN_ROWS = 1000


# ============================================================
# 1. Check file
# ============================================================

def check_file_exists() -> None:

    print("=" * 60)
    print("1. CHECKING DATASET FILE")
    print("=" * 60)

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print(f"Dataset found: {DATA_PATH}")


# ============================================================
# 2. Load dataset
# ============================================================

def load_dataset() -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("2. LOADING DATASET")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    return df


# ============================================================
# 3. Check empty dataset
# ============================================================

def check_empty(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("3. CHECKING EMPTY DATASET")
    print("=" * 60)

    if df.empty:

        raise ValueError(
            "Dataset is empty"
        )

    print("Dataset is not empty")


# ============================================================
# 4. Check schema
# ============================================================

def check_schema(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("4. CHECKING SCHEMA")
    print("=" * 60)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("Required columns found:")

    for column in REQUIRED_COLUMNS:

        print(f"  ✓ {column}")


# ============================================================
# 5. Check extra columns
# ============================================================

def check_extra_columns(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("5. CHECKING EXTRA COLUMNS")
    print("=" * 60)

    extra_columns = [
        column
        for column in df.columns
        if column not in REQUIRED_COLUMNS
    ]

    if extra_columns:

        print(
            f"Warning: Extra columns found: {extra_columns}"
        )

    else:

        print("No extra columns found")


# ============================================================
# 6. Check missing values
# ============================================================

def check_missing_values(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("6. CHECKING MISSING VALUES")
    print("=" * 60)

    missing = df[REQUIRED_COLUMNS].isnull().sum()

    total_missing = missing.sum()

    if total_missing > 0:

        print("Missing values:")

        print(
            missing[missing > 0]
        )

        raise ValueError(
            f"Dataset contains {total_missing} missing values"
        )

    print("✓ No missing values")


# ============================================================
# 7. Check duplicate rows
# ============================================================

def check_duplicates(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("7. CHECKING DUPLICATE ROWS")
    print("=" * 60)

    duplicates = df.duplicated().sum()

    print(
        f"Duplicate rows: {duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            f"Dataset contains {duplicates} duplicate rows"
        )

    print("✓ No duplicate rows")


# ============================================================
# 8. Check data types
# ============================================================

def check_data_types(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("8. CHECKING DATA TYPES")
    print("=" * 60)

    numeric_columns = [
        "amount",
        "account_age",
        "transaction_count",
        "fraud",
    ]

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            raise TypeError(
                f"Column '{column}' must be numeric"
            )

        print(
            f"✓ {column}: {df[column].dtype}"
        )


# ============================================================
# 9. Check amount
# ============================================================

def check_amount(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("9. CHECKING AMOUNT")
    print("=" * 60)

    if (df["amount"] < 0).any():

        raise ValueError(
            "Amount contains negative values"
        )

    print(
        f"Minimum amount: {df['amount'].min():.2f}"
    )

    print(
        f"Maximum amount: {df['amount'].max():.2f}"
    )

    print("✓ Amount values are valid")


# ============================================================
# 10. Check account age
# ============================================================

def check_account_age(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("10. CHECKING ACCOUNT AGE")
    print("=" * 60)

    if (df["account_age"] < 0).any():

        raise ValueError(
            "account_age contains negative values"
        )

    print(
        f"Minimum account age: {df['account_age'].min()}"
    )

    print(
        f"Maximum account age: {df['account_age'].max()}"
    )

    print("✓ Account age values are valid")


# ============================================================
# 11. Check transaction count
# ============================================================

def check_transaction_count(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("11. CHECKING TRANSACTION COUNT")
    print("=" * 60)

    if (df["transaction_count"] < 0).any():

        raise ValueError(
            "transaction_count contains negative values"
        )

    print(
        f"Minimum transactions: "
        f"{df['transaction_count'].min()}"
    )

    print(
        f"Maximum transactions: "
        f"{df['transaction_count'].max()}"
    )

    print("✓ Transaction counts are valid")


# ============================================================
# 12. Check fraud target
# ============================================================

def check_target(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("12. CHECKING FRAUD TARGET")
    print("=" * 60)

    unique_values = set(
        df["fraud"].unique()
    )

    print(
        f"Fraud values: {unique_values}"
    )

    allowed_values = {0, 1}

    invalid_values = (
        unique_values - allowed_values
    )

    if invalid_values:

        raise ValueError(
            f"Invalid fraud values: "
            f"{invalid_values}"
        )

    fraud_counts = df["fraud"].value_counts()

    print("\nClass distribution:")

    print(
        fraud_counts
    )

    print("✓ Fraud target is valid")


# ============================================================
# 13. Check infinite values
# ============================================================

def check_infinite_values(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 60)
    print("13. CHECKING INFINITE VALUES")
    print("=" * 60)

    numeric_df = df[
        REQUIRED_COLUMNS
    ].select_dtypes(
        include="number"
    )

    infinite_values = (
        numeric_df
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    if infinite_values > 0:

        raise ValueError(
            "Dataset contains infinite values"
        )

    print("✓ No infinite values")


# ============================================================
# 14. Check minimum dataset size
# ============================================================

def check_minimum_rows(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 60)
    print("14. CHECKING DATASET SIZE")
    print("=" * 60)

    if len(df) < MIN_ROWS:

        raise ValueError(
            f"Dataset has {len(df)} rows. "
            f"Minimum required: {MIN_ROWS}"
        )

    print(
        f"✓ Dataset contains {len(df)} rows"
    )


# ============================================================
# Main Validation
# ============================================================

def validate_dataset() -> None:

    print("\n")
    print("=" * 60)
    print("        DATA VALIDATION STARTED")
    print("=" * 60)

    check_file_exists()

    df = load_dataset()

    check_empty(df)

    check_schema(df)

    check_extra_columns(df)

    check_missing_values(df)

    check_duplicates(df)

    check_data_types(df)

    check_amount(df)

    check_account_age(df)

    check_transaction_count(df)

    check_target(df)

    check_infinite_values(df)

    check_minimum_rows(df)

    print("\n")
    print("=" * 60)
    print("        DATA VALIDATION PASSED ✓")
    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    try:

        validate_dataset()

    except Exception as error:

        print("\n")
        print("=" * 60)
        print("        DATA VALIDATION FAILED ✗")
        print("=" * 60)

        print(f"\nError: {error}")

        sys.exit(1)