import os
import numpy as np
import pandas as pd


def generate_data():

    np.random.seed(42)

    total_samples = 5000

    amount = np.random.uniform(
        10,
        10000,
        total_samples
    )

    account_age = np.random.randint(
        1,
        1000,
        total_samples
    )

    transaction_count = np.random.randint(
        1,
        100,
        total_samples
    )

    fraud = (
        (amount > 7000)
        & (account_age < 100)
    ).astype(int)

    df = pd.DataFrame({
        "amount": amount,
        "account_age": account_age,
        "transaction_count": transaction_count,
        "fraud": fraud
    })

    os.makedirs(
        "data/raw",
        exist_ok=True
    )

    df.to_csv(
        "data/source/transactions.csv",
        index=False
    )

    print("Dataset generated successfully.")

    print(df.head())

    print("\nClass distribution:")
    print(df["fraud"].value_counts())


if __name__ == "__main__":
    generate_data()