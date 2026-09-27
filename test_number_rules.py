import pandas as pd

print("=" * 70)
print("TESTING ADDRESS NUMBER RULES")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

print("\nLoading data...")

df = pd.read_csv("matching_features_v2_labeled_numbers.csv")

print(f"Loaded {len(df):,} rows")

# ---------------------------------------------------------
# 2. Current Rule 6 score
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

# Both addresses available
df.loc[both_address, "base_score"] = (
    0.50 * combined_name[both_address]
    + 0.50 * df.loc[both_address, "address_token_similarity"]
)

# Missing address
df.loc[~both_address, "base_score"] = (
    0.80 * combined_name[~both_address]
)

# ---------------------------------------------------------
# 3. Rule 6
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
# 4. Evaluation function
# ---------------------------------------------------------

def evaluate(prediction, label):

    tp = ((prediction == True) & (label == 1)).sum()
    fp = ((prediction == True) & (label == 0)).sum()
    fn = ((prediction == False) & (label == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    beta_squared = 0.5 ** 2

    f05 = (
        (1 + beta_squared)
        * precision
        * recall
        /
        (beta_squared * precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return tp, fp, fn, precision, recall, f05


label = df["label"]

# ---------------------------------------------------------
# 5. Test Rule 6
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BASELINE — RULE 6")
print("=" * 70)

result = evaluate(rule6, label)

print(
    f"TP={result[0]} "
    f"FP={result[1]} "
    f"FN={result[2]} "
    f"P={result[3]:.4f} "
    f"R={result[4]:.4f} "
    f"F0.5={result[5]:.4f}"
)

# ---------------------------------------------------------
# 6. Number rule experiments
# ---------------------------------------------------------

rules = {}

# Rule A
rules["A: number >= .50 + address >= .80"] = (
    rule6
    |
    (
        both_address
        & (df["number_overlap"] >= 0.50)
        & (df["address_similarity"] >= 0.80)
    )
)

# Rule B
rules["B: number >= .75 + address >= .80"] = (
    rule6
    |
    (
        both_address
        & (df["number_overlap"] >= 0.75)
        & (df["address_similarity"] >= 0.80)
    )
)

# Rule C
rules["C: exact number + address >= .80"] = (
    rule6
    |
    (
        both_address
        & (df["number_match"] == 1)
        & (df["address_similarity"] >= 0.80)
    )
)

# Rule D
rules["D: number >= .50 + address >= .90"] = (
    rule6
    |
    (
        both_address
        & (df["number_overlap"] >= 0.50)
        & (df["address_similarity"] >= 0.90)
    )
)

# Rule E
rules["E: number >= .75 + address >= .90"] = (
    rule6
    |
    (
        both_address
        & (df["number_overlap"] >= 0.75)
        & (df["address_similarity"] >= 0.90)
    )
)

# Rule F
rules["F: exact number + address >= .90"] = (
    rule6
    |
    (
        both_address
        & (df["number_match"] == 1)
        & (df["address_similarity"] >= 0.90)
    )
)

# ---------------------------------------------------------
# 7. Evaluate all rules
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NUMBER RULE RESULTS")
print("=" * 70)

results = []

for name, prediction in rules.items():

    result = evaluate(prediction, label)

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
# 8. Find best rule
# ---------------------------------------------------------

best = max(results, key=lambda x: x[6])

print("\n" + "=" * 70)
print("BEST NUMBER RULE")
print("=" * 70)

print(f"\n{best[0]}")

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