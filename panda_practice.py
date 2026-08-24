# import pandas as pd

# df =  pd.read_csv("data/raw/transactions.csv")
# # print(df.head())
# # print(df.shape)
# print(df.columns)
# # print("\n\n")
# # print("info is ",df.info)

# # df["account_age_years"] = df["account_age"] / 365
# # print("\n\n")
# # print("info is \n ",df.describe())


# # print(df.groupby("fraud")["transaction_count"].mean())
# # print(df.groupby("fraud")["amount"].mean())

# # print(df[df["amount"].isnull()])
# # print(df.duplicated().sum())
# # print(df["amount"].max())

# # print(df[df["amount"] > 9900])

# print(df[df["fraud"]==0].value_counts())