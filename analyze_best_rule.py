import pandas as pd


print("=" * 70)
print("ANALYZING BEST COMBINED RULE")
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
# BEST COMBINED RULE
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


prediction = (
    score >= 0.80
).astype(int)


# ============================================================
# FALSE NEGATIVES
# ============================================================

fn = df[
    (df["label"] == 1)
    &
    (prediction == 0)
].copy()


print()
print("=" * 70)
print("FALSE NEGATIVES")
print("=" * 70)

print(
    "Total:",
    len(fn)
)


# Calculate score for display

fn["score"] = score.loc[fn.index]


fn = fn.sort_values(
    "score",
    ascending=False
)


print()
print("Top 50 false negatives closest to threshold:")
print()


print(
    fn[
        [
            "source1_id",
            "matched_id",
            "source",
            "name_similarity",
            "name_token_similarity",
            "address_similarity",
            "address_token_similarity",
            "address1_present",
            "address2_present",
            "score"
        ]
    ].head(50).to_string(index=False)
)


# ============================================================
# FALSE POSITIVES
# ============================================================

fp = df[
    (df["label"] == 0)
    &
    (prediction == 1)
].copy()


print()
print("=" * 70)
print("FALSE POSITIVES")
print("=" * 70)

print(
    "Total:",
    len(fp)
)


fp["score"] = score.loc[fp.index]


fp = fp.sort_values(
    "score",
    ascending=False
)


print()
print("Top 30 false positives:")
print()


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
            "address1_present",
            "address2_present",
            "score"
        ]
    ].head(30).to_string(index=False)
)


# ============================================================
# THRESHOLD EXPERIMENT
# ============================================================

print()
print("=" * 70)
print("THRESHOLD EXPERIMENT")
print("=" * 70)
print()


def evaluate(threshold):

    pred = (
        score >= threshold
    ).astype(int)


    tp = (
        (df["label"] == 1)
        &
        (pred == 1)
    ).sum()


    fp = (
        (df["label"] == 0)
        &
        (pred == 1)
    ).sum()


    fn = (
        (df["label"] == 1)
        &
        (pred == 0)
    ).sum()


    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )


    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )


    f05 = (
        1.25 * precision * recall
        /
        (0.25 * precision + recall)
        if precision + recall > 0
        else 0
    )


    return (
        tp,
        fp,
        fn,
        precision,
        recall,
        f05
    )


for threshold in [
    0.70,
    0.72,
    0.74,
    0.76,
    0.78,
    0.80,
    0.82,
    0.84,
    0.86,
    0.88,
    0.90
]:

    result = evaluate(
        threshold
    )


    print(
        f"Threshold {threshold:.2f} "
        f"TP={result[0]:4d} "
        f"FP={result[1]:4d} "
        f"FN={result[2]:4d} "
        f"P={result[3]:.4f} "
        f"R={result[4]:.4f} "
        f"F0.5={result[5]:.4f}"
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)