import pandas as pd


print("=" * 70)
print("ANALYZING LARGE VALIDATION ERRORS")
print("=" * 70)


df = pd.read_csv(
    "matching_features_v2_labeled.csv"
)


# ============================================================
# FALSE NEGATIVES
# ============================================================

false_negatives = df[
    (df["label"] == 1)
    &
    (df["prediction"] == 0)
].copy()


print()
print("=" * 70)
print("FALSE NEGATIVES")
print("=" * 70)

print(
    "Total false negatives:",
    len(false_negatives)
)


# Sort by strongest address similarity
false_negatives = false_negatives.sort_values(
    [
        "address_token_similarity",
        "name_token_similarity"
    ],
    ascending=False
)


print()
print("Top 30 false negatives:")
print()

print(
    false_negatives[
        [
            "source1_id",
            "matched_id",
            "source",
            "name_similarity",
            "name_token_similarity",
            "address_similarity",
            "address_token_similarity",
            "address1_present",
            "address2_present"
        ]
    ].head(30).to_string(index=False)
)


# ============================================================
# FALSE POSITIVES
# ============================================================

false_positives = df[
    (df["label"] == 0)
    &
    (df["prediction"] == 1)
].copy()


print()
print("=" * 70)
print("FALSE POSITIVES")
print("=" * 70)

print(
    "Total false positives:",
    len(false_positives)
)


false_positives = false_positives.sort_values(
    [
        "name_token_similarity",
        "address_token_similarity"
    ],
    ascending=False
)


print()
print("Top 30 false positives:")
print()

print(
    false_positives[
        [
            "source1_id",
            "matched_id",
            "source",
            "name_similarity",
            "name_token_similarity",
            "address_similarity",
            "address_token_similarity",
            "address1_present",
            "address2_present"
        ]
    ].head(30).to_string(index=False)
)


# ============================================================
# ERROR COUNTS BY SOURCE
# ============================================================

print()
print("=" * 70)
print("ERRORS BY SOURCE")
print("=" * 70)


print()

print(
    "False negatives by source:"
)

print(
    false_negatives["source"].value_counts()
)


print()

print(
    "False positives by source:"
)

print(
    false_positives["source"].value_counts()
)


# ============================================================
# FALSE NEGATIVE PATTERNS
# ============================================================

print()
print("=" * 70)
print("FALSE NEGATIVE PATTERNS")
print("=" * 70)


def classify_fn(row):

    name = row["name_token_similarity"]

    address = row["address_token_similarity"]

    address_present = (
        row["address1_present"] == 1
        and row["address2_present"] == 1
    )


    if (
        name < 0.50
        and address >= 0.80
        and address_present
    ):
        return "weak name + strong address"

    if (
        name < 0.50
        and address < 0.80
        and address_present
    ):
        return "weak name + weak address"

    if (
        name >= 0.80
        and address < 0.50
        and address_present
    ):
        return "strong name + weak address"

    if (
        name >= 0.80
        and not address_present
    ):
        return "strong name + missing address"

    if (
        name >= 0.80
        and address >= 0.80
        and address_present
    ):
        return "strong name + strong address"

    if not address_present:
        return "weak name + missing address"

    return "other"


false_negatives["pattern"] = false_negatives.apply(
    classify_fn,
    axis=1
)


print()

print(
    false_negatives["pattern"].value_counts()
)


# ============================================================
# FALSE POSITIVE PATTERNS
# ============================================================

print()
print("=" * 70)
print("FALSE POSITIVE PATTERNS")
print("=" * 70)


def classify_fp(row):

    name = row["name_token_similarity"]

    address = row["address_token_similarity"]

    address_present = (
        row["address1_present"] == 1
        and row["address2_present"] == 1
    )


    if (
        name >= 0.90
        and not address_present
    ):
        return "very strong name + missing address"

    if (
        name >= 0.80
        and address < 0.50
        and address_present
    ):
        return "strong name + weak address"

    if (
        name < 0.80
        and address >= 0.80
        and address_present
    ):
        return "weak name + strong address"

    if not address_present:
        return "name-only / missing address"

    return "other"


false_positives["pattern"] = false_positives.apply(
    classify_fp,
    axis=1
)


print()

print(
    false_positives["pattern"].value_counts()
)


print()
print("=" * 70)
print("DONE")
print("=" * 70)