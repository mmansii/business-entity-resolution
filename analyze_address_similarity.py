import pandas as pd


print("=" * 70)
print("ANALYZING ADDRESS SIMILARITY")
print("=" * 70)


df = pd.read_csv(
    "matching_features_v2_labeled.csv"
)


# ============================================================
# BASIC STATISTICS
# ============================================================

positive = df[df["label"] == 1]
negative = df[df["label"] == 0]


print()
print("Positive pairs:", len(positive))
print("Negative pairs:", len(negative))


# ============================================================
# ADDRESS TOKEN SIMILARITY DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("ADDRESS TOKEN SIMILARITY")
print("=" * 70)


for threshold in [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    1.00
]:

    pos_count = (
        positive["address_token_similarity"]
        >= threshold
    ).sum()

    neg_count = (
        negative["address_token_similarity"]
        >= threshold
    ).sum()


    print(
        f">= {threshold:.2f} "
        f"| Positive: {pos_count:5d} "
        f"| Negative: {neg_count:7d}"
    )


# ============================================================
# ADDRESS CHARACTER SIMILARITY
# ============================================================

print()
print("=" * 70)
print("ADDRESS CHARACTER SIMILARITY")
print("=" * 70)


for threshold in [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    1.00
]:

    pos_count = (
        positive["address_similarity"]
        >= threshold
    ).sum()

    neg_count = (
        negative["address_similarity"]
        >= threshold
    ).sum()


    print(
        f">= {threshold:.2f} "
        f"| Positive: {pos_count:5d} "
        f"| Negative: {neg_count:7d}"
    )


# ============================================================
# BOTH ADDRESS FEATURES VERY HIGH
# ============================================================

print()
print("=" * 70)
print("VERY STRONG ADDRESS MATCHES")
print("=" * 70)


conditions = [

    (
        "token >= .90 AND similarity >= .70",
        (
            (df["address_token_similarity"] >= 0.90)
            &
            (df["address_similarity"] >= 0.70)
        )
    ),

    (
        "token >= .90 AND similarity >= .80",
        (
            (df["address_token_similarity"] >= 0.90)
            &
            (df["address_similarity"] >= 0.80)
        )
    ),

    (
        "token >= .95 AND similarity >= .80",
        (
            (df["address_token_similarity"] >= 0.95)
            &
            (df["address_similarity"] >= 0.80)
        )
    ),

    (
        "token = 1.00 AND similarity >= .80",
        (
            (df["address_token_similarity"] >= 1.00)
            &
            (df["address_similarity"] >= 0.80)
        )
    )
]


for description, condition in conditions:

    subset = df[condition]

    tp = (
        subset["label"] == 1
    ).sum()

    fp = (
        subset["label"] == 0
    ).sum()


    precision = (
        tp / len(subset)
        if len(subset) > 0
        else 0
    )


    print()
    print(description)

    print(
        "Total:",
        len(subset)
    )

    print(
        "Genuine:",
        tp
    )

    print(
        "False:",
        fp
    )

    print(
        "Precision:",
        round(precision, 4)
    )


# ============================================================
# FALSE POSITIVES WITH VERY HIGH ADDRESS SIMILARITY
# ============================================================

print()
print("=" * 70)
print("FALSE POSITIVES WITH VERY HIGH ADDRESS SIMILARITY")
print("=" * 70)


fp = df[
    (df["label"] == 0)
    &
    (df["address_token_similarity"] >= 0.90)
    &
    (df["address_similarity"] >= 0.70)
].copy()


print(
    "Count:",
    len(fp)
)


if len(fp) > 0:

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
                "address_token_similarity"
            ]
        ]
        .sort_values(
            "address_token_similarity",
            ascending=False
        )
        .head(50)
        .to_string(index=False)
    )


# ============================================================
# GENUINE MATCHES WITH WEAK NAME BUT STRONG ADDRESS
# ============================================================

print()
print("=" * 70)
print("GENUINE MATCHES: WEAK NAME + STRONG ADDRESS")
print("=" * 70)


weak_name_strong_address = df[
    (df["label"] == 1)
    &
    (df["name_token_similarity"] < 0.60)
    &
    (df["address_token_similarity"] >= 0.80)
].copy()


print(
    "Count:",
    len(weak_name_strong_address)
)


print()

print(
    weak_name_strong_address[
        [
            "source1_id",
            "matched_id",
            "source",
            "name_similarity",
            "name_token_similarity",
            "address_similarity",
            "address_token_similarity"
        ]
    ]
    .sort_values(
        "address_token_similarity",
        ascending=False
    )
    .head(50)
    .to_string(index=False)
)


print()
print("=" * 70)
print("DONE")
print("=" * 70)