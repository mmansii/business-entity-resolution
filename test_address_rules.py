import pandas as pd

print("=" * 70)
print("TESTING ADDRESS + NUMBER COMBINATIONS")
print("=" * 70)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("\nLoading data...")

df = pd.read_csv(
    "matching_features_v2_labeled_numbers.csv"
)

print(f"Loaded {len(df):,} rows")

# ---------------------------------------------------------
# Current Rule 6
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
# Current best Rule B
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
# Evaluation function
# ---------------------------------------------------------

def evaluate(prediction):

    label = df["label"]

    tp = ((prediction == True) & (label == 1)).sum()
    fp = ((prediction == True) & (label == 0)).sum()
    fn = ((prediction == False) & (label == 1)).sum()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    beta2 = 0.25

    f05 = (
        1.25 * precision * recall
        /
        (0.25 * precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return tp, fp, fn, precision, recall, f05


# ---------------------------------------------------------
# Baseline
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BASELINES")
print("=" * 70)

result = evaluate(rule6)

print(
    f"Rule 6: "
    f"TP={result[0]} "
    f"FP={result[1]} "
    f"FN={result[2]} "
    f"P={result[3]:.4f} "
    f"R={result[4]:.4f} "
    f"F0.5={result[5]:.4f}"
)

result = evaluate(rule_b)

print(
    f"Rule B:  "
    f"TP={result[0]} "
    f"FP={result[1]} "
    f"FN={result[2]} "
    f"P={result[3]:.4f} "
    f"R={result[4]:.4f} "
    f"F0.5={result[5]:.4f}"
)

# ---------------------------------------------------------
# New rules
# ---------------------------------------------------------

rules = {}

# G
rules["G: address .90 + token .90"] = (
    rule_b
    |
    (
        both_address
        & (df["address_similarity"] >= 0.90)
        & (df["address_token_similarity"] >= 0.90)
    )
)

# H
rules["H: address .80 + token .90 + number .50"] = (
    rule_b
    |
    (
        both_address
        & (df["address_similarity"] >= 0.80)
        & (df["address_token_similarity"] >= 0.90)
        & (df["number_overlap"] >= 0.50)
    )
)

# I
rules["I: address .95 + token .95"] = (
    rule_b
    |
    (
        both_address
        & (df["address_similarity"] >= 0.95)
        & (df["address_token_similarity"] >= 0.95)
    )
)

# J
rules["J: address .85 + token .85 + name .50"] = (
    rule_b
    |
    (
        both_address
        & (df["address_similarity"] >= 0.85)
        & (df["address_token_similarity"] >= 0.85)
        & (df["name_token_similarity"] >= 0.50)
    )
)

# K
rules["K: address .90 + token .90 + name .40"] = (
    rule_b
    |
    (
        both_address
        & (df["address_similarity"] >= 0.90)
        & (df["address_token_similarity"] >= 0.90)
        & (df["name_token_similarity"] >= 0.40)
    )
)

# L
rules["L: address .85 + token .90 + number .50"] = (
    rule_b
    |
    (
        both_address
        & (df["address_similarity"] >= 0.85)
        & (df["address_token_similarity"] >= 0.90)
        & (df["number_overlap"] >= 0.50)
    )
)

# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NEW ADDRESS RULE RESULTS")
print("=" * 70)

results = []

for name, prediction in rules.items():

    result = evaluate(prediction)

    results.append(
        (
            name,
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
            result[5]
        )
    )

    print(f"\n{name}")

    print(
        f"TP={result[0]} "
        f"FP={result[1]} "
        f"FN={result[2]} "
        f"P={result[3]:.4f} "
        f"R={result[4]:.4f} "
        f"F0.5={result[5]:.4f}"
    )

# ---------------------------------------------------------
# Best
# ---------------------------------------------------------

best = max(
    results,
    key=lambda x: x[6]
)

print("\n" + "=" * 70)
print("BEST ADDRESS RULE")
print("=" * 70)

print(best[0])

print(
    f"TP={best[1]} "
    f"FP={best[2]} "
    f"FN={best[3]} "
    f"P={best[4]:.4f} "
    f"R={best[5]:.4f} "
    f"F0.5={best[6]:.4f}"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)