import pandas as pd

print("=" * 70)
print("ANALYZING REMAINING FALSE NEGATIVES")
print("=" * 70)

df = pd.read_csv("matching_features_v2_labeled_numbers.csv")

# ---------------------------------------------------------
# Calculate Rule 6 baseline score
# ---------------------------------------------------------

combined_name = (
    0.40 * df["name_similarity"]
    + 0.60 * df["name_token_similarity"]
)

both_address = (
    (df["address1_present"] == 1)
    & (df["address2_present"] == 1)
)

df["base_score"] = 0.0

df.loc[both_address, "base_score"] = (
    0.50 * combined_name[both_address]
    + 0.50 * df.loc[
        both_address,
        "address_token_similarity"
    ]
)

df.loc[~both_address, "base_score"] = (
    0.80 * combined_name[~both_address]
)

# ---------------------------------------------------------
# Rule 6
# ---------------------------------------------------------

rule6 = (
    (df["base_score"] >= 0.80)
    |
    (
        both_address
        & (df["name_token_similarity"] >= 0.95)
        & (df["address_token_similarity"] >= 0.55)
        & (df["base_score"] >= 0.72)
    )
)

# ---------------------------------------------------------
# Rule B
# ---------------------------------------------------------

rule_b = (
    rule6
    |
    (
        both_address
        & (df["number_overlap"] >= 0.75)
        & (df["address_similarity"] >= 0.80)
    )
)

# ---------------------------------------------------------
# Remaining false negatives
# ---------------------------------------------------------

false_negatives = df[
    (df["label"] == 1)
    & (~rule_b)
].copy()

print("\nRemaining genuine matches missed by Rule B:")
print(len(false_negatives))

# ---------------------------------------------------------
# Categorize them
# ---------------------------------------------------------

def categorize(row):

    name_token = row["name_token_similarity"]
    name_sim = row["name_similarity"]

    addr_token = row["address_token_similarity"]
    addr_sim = row["address_similarity"]

    number = row["number_overlap"]

    address_present = (
        row["address1_present"] == 1
        and row["address2_present"] == 1
    )

    if not address_present:
        if name_token >= 0.90:
            return "strong name + missing address"
        elif name_token >= 0.70:
            return "medium name + missing address"
        else:
            return "weak name + missing address"

    if number >= 0.75 and addr_sim < 0.80:
        return "strong numbers + weak address similarity"

    if addr_sim >= 0.80 and addr_token >= 0.75:
        if name_token >= 0.70:
            return "strong address + good name"
        else:
            return "strong address + weak name"

    if name_token >= 0.90:
        return "strong name + weak address"

    if name_token >= 0.70:
        return "medium name + weak address"

    return "weak name + weak address"


false_negatives["category"] = false_negatives.apply(
    categorize,
    axis=1
)

print("\n" + "=" * 70)
print("FALSE NEGATIVE CATEGORIES")
print("=" * 70)

print(
    false_negatives["category"]
    .value_counts()
    .to_string()
)

# ---------------------------------------------------------
# Score ranges
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FALSE NEGATIVE SCORE RANGES")
print("=" * 70)

bins = [
    0.0,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    1.00
]

labels = [
    "< .30",
    ".30-.40",
    ".40-.50",
    ".50-.60",
    ".60-.70",
    ".70-.75",
    ".75-.80",
    ".80+"
]

false_negatives["score_range"] = pd.cut(
    false_negatives["base_score"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print(
    false_negatives["score_range"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ---------------------------------------------------------
# Strong address but missed
# ---------------------------------------------------------

strong_address_missed = false_negatives[
    (false_negatives["address_similarity"] >= 0.80)
    |
    (false_negatives["address_token_similarity"] >= 0.90)
]

print("\n" + "=" * 70)
print("MISSED MATCHES WITH STRONG ADDRESS")
print("=" * 70)

print(
    f"Count: {len(strong_address_missed)}"
)

columns = [
    "source1_id",
    "matched_id",
    "source",
    "name_similarity",
    "name_token_similarity",
    "address_similarity",
    "address_token_similarity",
    "number_overlap",
    "number_match",
    "base_score"
]

print(
    strong_address_missed[
        columns
    ]
    .sort_values(
        ["address_similarity"],
        ascending=False
    )
    .head(50)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)