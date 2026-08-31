
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# Configuration
# ============================================================

RAW_DATA_PATH = Path("data/raw/transactions.csv")

PROCESSED_DATA_DIR = Path("data/processed")

FEATURE_COLUMNS = [
    "amount",
    "account_age",
    "transaction_count",
]

TARGET_COLUMN = "fraud"

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# Load Data
# ============================================================


def load_data() -> pd.DataFrame:
    """Load validated raw dataset."""

    print("=" * 60)
    print("1. LOADING RAW DATA")
    print("=" * 60)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {list(df.columns)}")

    return df


# ============================================================
# Remove Duplicates
# ============================================================


def remove_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove duplicate rows."""

    print("\n" + "=" * 60)
    print("2. REMOVING DUPLICATES")
    print("=" * 60)

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    removed = before - after

    print(f"Rows before : {before}")
    print(f"Rows after  : {after}")
    print(f"Removed     : {removed}")

    return df


# ============================================================
# Select Features and Target
# ============================================================


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target."""

    print("\n" + "=" * 60)
    print("3. PREPARING FEATURES AND TARGET")
    print("=" * 60)

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN]

    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target  : {TARGET_COLUMN}")
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")

    return X, y


# ============================================================
# Train Test Split
# ============================================================


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Split dataset using stratification."""

    print("\n" + "=" * 60)
    print("4. TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# Feature Scaling
# ============================================================


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    StandardScaler,
]:
    """Scale numerical features."""

    print("\n" + "=" * 60)
    print("5. FEATURE SCALING")
    print("=" * 60)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=FEATURE_COLUMNS,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=FEATURE_COLUMNS,
        index=X_test.index,
    )

    print("✓ StandardScaler applied")

    return (
        X_train_scaled,
        X_test_scaled,
        scaler,
    )


# ============================================================
# Save Data
# ============================================================


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
) -> None:
    """Save processed datasets and scaler."""

    print("\n" + "=" * 60)
    print("6. SAVING PROCESSED DATA")
    print("=" * 60)

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train_path = (
        PROCESSED_DATA_DIR
        / "X_train.csv"
    )

    X_test_path = (
        PROCESSED_DATA_DIR
        / "X_test.csv"
    )

    y_train_path = (
        PROCESSED_DATA_DIR
        / "y_train.csv"
    )

    y_test_path = (
        PROCESSED_DATA_DIR
        / "y_test.csv"
    )

    scaler_path = (
        PROCESSED_DATA_DIR
        / "scaler.pkl"
    )

    X_train.to_csv(
        X_train_path,
        index=False,
    )

    X_test.to_csv(
        X_test_path,
        index=False,
    )

    y_train.to_csv(
        y_train_path,
        index=False,
    )

    y_test.to_csv(
        y_test_path,
        index=False,
    )

    with open(
        scaler_path,
        "wb",
    ) as file:
        pickle.dump(
            scaler,
            file,
        )

    print(f"✓ {X_train_path}")
    print(f"✓ {X_test_path}")
    print(f"✓ {y_train_path}")
    print(f"✓ {y_test_path}")
    print(f"✓ {scaler_path}")


# ============================================================
# Main Preprocessing Pipeline
# ============================================================


def preprocess() -> None:
    """Run the complete data preprocessing pipeline."""

    print("\n")
    print("=" * 60)
    print("       DATA PREPROCESSING STARTED")
    print("=" * 60)

    # 1. Load
    df = load_data()

    # 2. Remove duplicates
    df = remove_duplicates(df)

    # 3. Features / target
    X, y = prepare_features(df)

    # 4. Train/test split
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(X, y)

    # 5. Scaling
    (
        X_train_scaled,
        X_test_scaled,
        scaler,
    ) = scale_features(
        X_train,
        X_test,
    )

    # 6. Save
    save_processed_data(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
    )

    print("\n")
    print("=" * 60)
    print("       DATA PREPROCESSING COMPLETED ✓")
    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    try:
        preprocess()

    except Exception as error:  # noqa: BLE001
        print("\n")
        print("=" * 60)
        print("       DATA PREPROCESSING FAILED ✗")
        print("=" * 60)

        print(f"\nError: {error}")

        sys.exit(1)

