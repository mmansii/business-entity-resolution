import pandas as pd


print("=" * 70)
print("ANALYZING CANDIDATE RANKING")
print("=" * 70)


df = pd.read_csv(
    "matching_features_v2_labeled.csv"
)


name = df["name_similarity"]

name_token = df["name_token_similarity"]

address_token = df["address_token_similarity"]


both_address = (
    (df["address1_present"] == 1)
    &
    (df["address2_present"] == 1)
)


# ============================================================
# BASE SCORE
# ============================================================

combined_name = (
    0.40 * name
    +
    0.60 * name_token
)


score = pd.Series(
    0.80 * combined_name,
    index=df.index
)


score.loc[both_address] = (
    0.50 * combined_name[both_address]
    +
    0.50 * address_token[both_address]
)


df["score"] = score


# ============================================================
# RANK CANDIDATES FOR EACH SOURCE 1 ENTITY
# ============================================================

df["rank"] = (
    df.groupby("source1_id")["score"]
      .rank(
          method="first",
          ascending=False
      )
)


# ============================================================
# TOP RANK DISTRIBUTION OF GENUINE MATCHES
# ============================================================

print()
print("=" * 70)
print("GENUINE MATCH RANK DISTRIBUTION")
print("=" * 70)


positive = df[df["label"] == 1]


for r in range(1, 11):

    count = (
        positive["rank"] == r
    ).sum()

    print(
        f"Rank {r}: {count}"
    )


print()
print(
    "Genuine matches with rank <= 5:",
    (positive["rank"] <= 5).sum()
)

print(
    "Total genuine matches:",
    len(positive)
)


# ============================================================
# FALSE NEGATIVES THAT ARE HIGHLY RANKED
# ============================================================

false_negative = (
    (df["label"] == 1)
    &
    (df["score"] < 0.80)
)


fn = df[false_negative].copy()


print()
print("=" * 70)
print("FALSE NEGATIVES WITH HIGH RANK")
print("=" * 70)


fn_high_rank = fn[
    fn["rank"] <= 3
].sort_values(
    ["rank", "score"],
    ascending=[True, False]
)


print(
    fn_high_rank[
        [
            "source1_id",
            "matched_id",
            "source",
            "score",
            "rank",
            "name_similarity",
            "name_token_similarity",
            "address_token_similarity"
        ]
    ].head(50).to_string(index=False)
)


# ============================================================
# FALSE NEGATIVE RANK SUMMARY
# ============================================================

print()
print("=" * 70)
print("FALSE NEGATIVE RANK SUMMARY")
print("=" * 70)


for r in [1, 2, 3, 5, 10]:

    count = (
        fn["rank"] <= r
    ).sum()

    print(
        f"False negatives with rank <= {r}: {count}"
    )


# ============================================================
# SCORE VS RANK
# ============================================================

print()
print("=" * 70)
print("FALSE NEGATIVES BY SCORE RANGE")
print("=" * 70)


ranges = [
    (0.70, 0.72),
    (0.72, 0.74),
    (0.74, 0.76),
    (0.76, 0.78),
    (0.78, 0.80)
]


for low, high in ranges:

    subset = fn[
        (fn["score"] >= low)
        &
        (fn["score"] < high)
    ]

    print(
        f"{low:.2f}-{high:.2f}: "
        f"{len(subset)} total, "
        f"{(subset['rank'] <= 3).sum()} rank<=3"
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)