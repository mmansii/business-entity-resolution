import pandas as pd

print("=" * 70)
print("FINDING V2 BLOCKING MISSES")
print("=" * 70)

print()
print("Loading ground truth...")

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=1000
)

source1_ids = set(source1["entity_id"])

ground_truth = ground_truth[
    ground_truth["source1_entity_id"].isin(source1_ids)
]

print("Loading V2 candidate pairs...")

candidates = pd.read_csv(
    "blocked_validation_candidates_v2.csv"
)

candidate_pairs = set(
    zip(
        candidates["source1_id"],
        candidates["matched_id"]
    )
)

misses = []

for _, row in ground_truth.iterrows():

    source1_id = row["source1_entity_id"]

    if pd.isna(row["matched_entity_ids"]):
        continue

    matched_ids = str(
        row["matched_entity_ids"]
    ).split(",")

    for matched_id in matched_ids:

        matched_id = matched_id.strip()

        if not matched_id:
            continue

        if (
            source1_id,
            matched_id
        ) not in candidate_pairs:

            misses.append({
                "source1_id": source1_id,
                "matched_id": matched_id
            })


misses_df = pd.DataFrame(misses)

misses_df.to_csv(
    "blocking_misses_v2.csv",
    index=False
)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print()
print("Missed matches:", len(misses_df))

if len(misses_df) > 0:

    print()
    print(misses_df.to_string(index=False))

print()
print("Saved:")
print("blocking_misses_v2.csv")