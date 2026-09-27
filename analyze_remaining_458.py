import pandas as pd

print("=" * 70)
print("DEEP ANALYSIS OF REMAINING FALSE NEGATIVES")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

print("\nLoading data...")

df = pd.read_csv("matching_features_v2_labeled_numbers.csv")

print(f"Loaded {len(df):,} rows")

# ---------------------------------------------------------
# 2. Calculate Rule B + Rule I
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

# Rule 6
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

# Rule B
rule_b = (
    rule6
    |
    (
        both_address
        & (df["number_overlap"] >= 0.75)
        & (df["address_similarity"] >= 0.80)
    )
)

# Rule I
rule_i = (
    rule_b
    |
    (
        both_address
        & (df["address_similarity"] >= 0.95)
        & (df["address_token_similarity"] >= 0.95)
    )
)

# ---------------------------------------------------------
# 3. Get remaining false negatives
# ---------------------------------------------------------

fn = df[
    (df["label"] == 1)
    & (~rule_i)
].copy()

print("\n" + "=" * 70)
print("REMAINING FALSE NEGATIVES")
print("=" * 70)

print(f"\nTotal remaining FNs: {len(fn)}")

# ---------------------------------------------------------
# 4. Source breakdown
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FALSE NEGATIVES BY SOURCE")
print("=" * 70)

print(
    fn["source"]
    .value_counts()
    .to_string()
)

# ---------------------------------------------------------
# 5. Name strength
# ---------------------------------------------------------

def name_category(x):

    if x >= 0.90:
        return "very strong name >= .90"

    if x >= 0.80:
        return "strong name .80-.90"

    if x >= 0.70:
        return "good name .70-.80"

    if x >= 0.50:
        return "medium name .50-.70"

    if x >= 0.30:
        return "weak name .30-.50"

    return "very weak name < .30"


fn["name_category"] = fn[
    "name_token_similarity"
].apply(name_category)

print("\n" + "=" * 70)
print("NAME STRENGTH")
print("=" * 70)

print(
    fn["name_category"]
    .value_counts()
    .to_string()
)

# ---------------------------------------------------------
# 6. Address strength
# ---------------------------------------------------------

def address_category(row):

    if row["address1_present"] == 0 or row["address2_present"] == 0:
        return "missing address"

    if (
        row["address_similarity"] >= 0.90
        and row["address_token_similarity"] >= 0.90
    ):
        return "very strong address"

    if (
        row["address_similarity"] >= 0.80
        and row["address_token_similarity"] >= 0.80
    ):
        return "strong address"

    if (
        row["address_similarity"] >= 0.60
        and row["address_token_similarity"] >= 0.60
    ):
        return "medium address"

    return "weak address"


fn["address_category"] = fn.apply(
    address_category,
    axis=1
)

print("\n" + "=" * 70)
print("ADDRESS STRENGTH")
print("=" * 70)

print(
    fn["address_category"]
    .value_counts()
    .to_string()
)

# ---------------------------------------------------------
# 7. Number strength
# ---------------------------------------------------------

def number_category(x):

    if x >= 1.0:
        return "exact numbers"

    if x >= 0.75:
        return "strong numbers .75-<1"

    if x >= 0.50:
        return "medium numbers .50-.75"

    if x > 0:
        return "weak numbers 0-.50"

    return "no number overlap"


fn["number_category"] = fn[
    "number_overlap"
].apply(number_category)

print("\n" + "=" * 70)
print("NUMBER STRENGTH")
print("=" * 70)

print(
    fn["number_category"]
    .value_counts()
    .to_string()
)

# ---------------------------------------------------------
# 8. Missing addresses
# ---------------------------------------------------------

missing_address = (
    (fn["address1_present"] == 0)
    |
    (fn["address2_present"] == 0)
)

print("\n" + "=" * 70)
print("MISSING ADDRESS")
print("=" * 70)

print(
    f"Missing address: {missing_address.sum()}"
)

print(
    f"Both addresses present: {(~missing_address).sum()}"
)

# ---------------------------------------------------------
# 9. Strong name but missed
# ---------------------------------------------------------

strong_name = fn[
    fn["name_token_similarity"] >= 0.90
]

print("\n" + "=" * 70)
print("VERY STRONG NAME BUT STILL MISSED")
print("=" * 70)

print(
    f"Count: {len(strong_name)}"
)

cols = [
    "source1_id",
    "matched_id",
    "source",
    "name_similarity",
    "name_token_similarity",
    "address_similarity",
    "address_token_similarity",
    "number_overlap",
    "base_score"
]

print(
    strong_name[
        cols
    ]
    .sort_values(
        "name_token_similarity",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)

# ---------------------------------------------------------
# 10. Strong numbers but missed
# ---------------------------------------------------------

strong_numbers = fn[
    fn["number_overlap"] >= 0.50
]

print("\n" + "=" * 70)
print("STRONG NUMBER AGREEMENT BUT STILL MISSED")
print("=" * 70)

print(
    f"Count: {len(strong_numbers)}"
)

print(
    strong_numbers[
        cols
    ]
    .sort_values(
        "number_overlap",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)

# ---------------------------------------------------------
# 11. Strong address but missed
# ---------------------------------------------------------

strong_address = fn[
    (fn["address_similarity"] >= 0.80)
    &
    (fn["address_token_similarity"] >= 0.80)
]

print("\n" + "=" * 70)
print("STRONG ADDRESS BUT STILL MISSED")
print("=" * 70)

print(
    f"Count: {len(strong_address)}"
)

print(
    strong_address[
        cols
    ]
    .sort_values(
        "address_similarity",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)

# ---------------------------------------------------------
# 12. Very strong name OR address
# ---------------------------------------------------------

easy_candidates = fn[
    (fn["name_token_similarity"] >= 0.80)
    |
    (
        (fn["address_similarity"] >= 0.90)
        &
        (fn["address_token_similarity"] >= 0.90)
    )
]

print("\n" + "=" * 70)
print("POTENTIALLY EASY MISSES")
print("=" * 70)

print(
    f"Count: {len(easy_candidates)}"
)

# ---------------------------------------------------------
# 13. Hard cases
# ---------------------------------------------------------

hard_cases = fn[
    (fn["name_token_similarity"] < 0.50)
    &
    (fn["address_similarity"] < 0.60)
]

print("\n" + "=" * 70)
print("VERY HARD CASES")
print("=" * 70)

print(
    f"Count: {len(hard_cases)}"
)

# ---------------------------------------------------------
# 14. Score distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BASE SCORE DISTRIBUTION")
print("=" * 70)

bins = [
    0.0,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80
]

labels = [
    "< .30",
    ".30-.40",
    ".40-.50",
    ".50-.60",
    ".60-.70",
    ".70-.75",
    ".75-.80"
]

fn["score_range"] = pd.cut(
    fn["base_score"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print(
    fn["score_range"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ---------------------------------------------------------
# 15. Save analysis
# ---------------------------------------------------------

fn.to_csv(
    "remaining_false_negatives.csv",
    index=False
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "\nSaved detailed results to:"
)

print(
    "remaining_false_negatives.csv"
)