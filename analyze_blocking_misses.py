import pandas as pd


print("=" * 70)
print("ANALYZING BLOCKING MISSES")
print("=" * 70)


# =========================================================
# LOAD DATA
# =========================================================

print()
print("Loading source 1...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
).head(1000)


print("Loading ground truth...")

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

gt = gt[
    gt["source1_entity_id"].isin(
        source1["entity_id"]
    )
]


print("Loading blocked candidates...")

blocked = pd.read_csv(
    "blocked_validation_candidates.csv",
    usecols=[
        "source1_id",
        "matched_id"
    ]
)


# =========================================================
# BUILD GROUND TRUTH PAIRS
# =========================================================

print()
print("Building genuine match list...")

true_pairs = []

for _, row in gt.iterrows():

    source1_id = row["source1_entity_id"]

    if pd.isna(row["matched_entity_ids"]):
        continue

    matched_ids = str(
        row["matched_entity_ids"]
    ).split(",")

    for matched_id in matched_ids:

        matched_id = matched_id.strip()

        if matched_id:

            true_pairs.append(
                (
                    source1_id,
                    matched_id
                )
            )


true_pairs = set(true_pairs)


# =========================================================
# BLOCKED PAIRS
# =========================================================

blocked_pairs = set(
    zip(
        blocked["source1_id"],
        blocked["matched_id"]
    )
)


# =========================================================
# FIND MISSES
# =========================================================

missed = true_pairs - blocked_pairs


print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    "Total genuine pairs:",
    len(true_pairs)
)

print(
    "Found by blocking:",
    len(true_pairs & blocked_pairs)
)

print(
    "Missed by blocking:",
    len(missed)
)


# =========================================================
# SHOW MISSED IDS
# =========================================================

print()
print("=" * 70)
print("MISSED GENUINE MATCHES")
print("=" * 70)

for source1_id, matched_id in sorted(missed):

    print(
        source1_id,
        "->",
        matched_id
    )


# =========================================================
# SAVE
# =========================================================

missed_df = pd.DataFrame(
    list(missed),
    columns=[
        "source1_id",
        "matched_id"
    ]
)

missed_df.to_csv(
    "blocking_misses.csv",
    index=False
)


print()
print("=" * 70)
print("Saved:")
print("blocking_misses.csv")
print("=" * 70)