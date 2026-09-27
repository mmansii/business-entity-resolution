import pandas as pd


print("=" * 70)
print("ANALYZING RULE 6")
print("=" * 70)


df = pd.read_csv(
    "matching_features_v2_labeled.csv"
)


name = df["name_similarity"]

name_token = df["name_token_similarity"]

address = df["address_similarity"]

address_token = df["address_token_similarity"]


both_address = (
    (df["address1_present"] == 1)
    &
    (df["address2_present"] == 1)
)


# ============================================================
# CALCULATE BASE SCORE
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


# ============================================================
# RULE 6
# ============================================================

prediction = (
    score >= 0.80
)


special_rule = (
    both_address
    &
    (name_token >= 0.95)
    &
    (address_token >= 0.55)
    &
    (score >= 0.72)
)


prediction = prediction | special_rule


# ============================================================
# FALSE POSITIVES CREATED BY RULE 6
# ============================================================

false_positive = (
    (df["label"] == 0)
    &
    prediction
)


print()
print("=" * 70)
print("FALSE POSITIVES")
print("=" * 70)


fp = df[false_positive].copy()

fp["base_score"] = score[false_positive].values

fp["triggered_rule6"] = special_rule[false_positive].values

fp = fp.sort_values(
    "base_score",
    ascending=False
)


print(
    fp[
        [
            "source1_id",
            "matched_id",
            "source",
            "name_similarity",
            "name_token_similarity",
            "address_similarity",
            "address_token_similarity",
            "base_score",
            "triggered_rule6"
        ]
    ].to_string(index=False)
)


# ============================================================
# MATCHES RECOVERED ONLY BY RULE 6
# ============================================================

recovered = (
    (df["label"] == 1)
    &
    special_rule
    &
    (score < 0.80)
)


print()
print("=" * 70)
print("GENUINE MATCHES RECOVERED ONLY BY RULE 6")
print("=" * 70)

recovered_df = df[recovered].copy()

recovered_df["base_score"] = score[recovered].values

recovered_df = recovered_df.sort_values(
    "base_score",
    ascending=False
)


print(
    recovered_df[
        [
            "source1_id",
            "matched_id",
            "source",
            "name_similarity",
            "name_token_similarity",
            "address_similarity",
            "address_token_similarity",
            "base_score"
        ]
    ].head(50).to_string(index=False)
)


# ============================================================
# HOW MANY RULE 6 MATCHES COME FROM DIFFERENT SCORE RANGES?
# ============================================================

print()
print("=" * 70)
print("RULE 6 RECOVERED MATCH SCORE DISTRIBUTION")
print("=" * 70)


ranges = [
    (0.72, 0.74),
    (0.74, 0.76),
    (0.76, 0.78),
    (0.78, 0.80)
]


for low, high in ranges:

    count = (
        recovered
        &
        (score >= low)
        &
        (score < high)
    ).sum()

    print(
        f"{low:.2f} - {high:.2f}: {count}"
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)