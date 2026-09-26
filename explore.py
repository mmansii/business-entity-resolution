import pandas as pd

path = "dataset/train/train_ground_truth.tsv"

df = pd.read_csv(path, sep="\t", nrows=5)

print("First 5 rows:")
print(df)

print("\nColumns:")
print(df.columns.tolist())