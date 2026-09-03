
import shutil
import sys
from pathlib import Path

import pandas as pd

# ============================================================
# Configuration
# ============================================================

SOURCE_PATH = Path("data/raw/transactions.csv")

RAW_DATA_DIR = Path("data/raw")

RAW_DATA_PATH = RAW_DATA_DIR / "transactions.csv"


# ============================================================
# Check Source
# ============================================================


def check_source_file() -> None:
    """Check whether source dataset exists."""

    print("=" * 60)
    print("1. CHECKING SOURCE DATA")
    print("=" * 60)

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_PATH}"
        )

    print(
        f"Source dataset found: {SOURCE_PATH}"
    )


# ============================================================
# Create Raw Data Directory
# ============================================================


def create_raw_directory() -> None:
    """Create raw data directory if it does not exist."""

    print("\n" + "=" * 60)
    print("2. CREATING RAW DATA DIRECTORY")
    print("=" * 60)

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Raw data directory: {RAW_DATA_DIR}"
    )


# ============================================================
# Ingest Data
# ============================================================


def ingest_data() -> None:
    """Copy source dataset into raw data directory."""

    print("\n" + "=" * 60)
    print("3. INGESTING DATA")
    print("=" * 60)

    shutil.copy2(
        SOURCE_PATH,
        RAW_DATA_PATH,
    )

    print(
        f"Data copied to: {RAW_DATA_PATH}"
    )


# ============================================================
# Verify Ingested Data
# ============================================================


def verify_data() -> None:
    """Verify ingested dataset."""

    print("\n" + "=" * 60)
    print("4. VERIFYING INGESTED DATA")
    print("=" * 60)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Ingested dataset was not created"
        )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print(
        f"Columns : {list(df.columns)}"
    )

    if df.empty:
        raise ValueError(
            "Ingested dataset is empty"
        )

    print(
        "✓ Ingested dataset verified"
    )


# ============================================================
# Main Pipeline
# ============================================================


def ingest() -> None:
    """Run the complete data ingestion pipeline."""

    print("\n")
    print("=" * 60)
    print("          DATA INGESTION STARTED")
    print("=" * 60)

    check_source_file()

    create_raw_directory()

    ingest_data()

    verify_data()

    print("\n")
    print("=" * 60)
    print("          DATA INGESTION COMPLETED ✓")
    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================


if __name__ == "__main__":
    try:
        ingest()

    except Exception as error:  # noqa: BLE001
        print("\n")
        print("=" * 60)
        print("          DATA INGESTION FAILED ✗")
        print("=" * 60)

        print(f"\nError: {error}")

        sys.exit(1)

