import pandas as pd

files = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv"
]

for file in files:
    print("\n" + "=" * 60)
    print(file)

    df = pd.read_csv(file, sep="\t")

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())