import pandas as pd

files = [
    "dataset/test/test_source1.tsv",
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv",
]

for file in files:
    print(f"\nReading: {file}")

    df = pd.read_csv(
        file,
        sep="\t",
        usecols=["entity_id"]
    )

    print(f"Rows: {len(df):,}")