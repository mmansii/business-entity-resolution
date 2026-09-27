import pandas as pd

df = pd.read_csv("matching_features_v2_labeled_numbers.csv")

# ---------------------------------------------------------
# Common calculations
# ---------------------------------------------------------

combined_name = (
    0.40 * df["name_similarity"]
    + 0.60 * df["name_token_similarity"]
)

both_address = (
    (df["address1_present"] == 1)
    & (df["address2_present"] == 1)
)

base_score = pd.Series(0.0, index=df.index)

base_score.loc[both_address] = (
    0.50 * combined_name[both_address]
    + 0.50 * df.loc[
        both_address,
        "address_token_similarity"
    ]
)

base_score.loc[~both_address] = (
    0.80 * combined_name[~both_address]
)

# ---------------------------------------------------------
# Existing Rule I
# ---------------------------------------------------------

rule6 = (
    (base_score >= 0.80)
    |
    (
        both_address
        & (df["name_token_similarity"] >= 0.95)
        & (df["address_token_similarity"] >= 0.55)
        & (base_score >= 0.72)
    )
)

rule_b = (
    rule6
    |
    (
        both_address
        & (df["number_overlap"] >= 0.75)
        & (df["address_similarity"] >= 0.80)
    )
)

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
# Evaluation function
# ---------------------------------------------------------

def evaluate(name, prediction):

    tp = ((prediction) & (df["label"] == 1)).sum()
    fp = ((prediction) & (df["label"] == 0)).sum()
    fn = ((~prediction) & (df["label"] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f05 = (
        1.25 * precision * recall
        / (0.25 * precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(
        f"{name:<45}"
        f" TP={tp:<5}"
        f" FP={fp:<5}"
        f" FN={fn:<5}"
        f" P={precision:.4f}"
        f" R={recall:.4f}"
        f" F0.5={f05:.4f}"
    )


print("\n" + "=" * 115)
print("BASELINE + TARGETED RULE EXPERIMENTS")
print("=" * 115)

evaluate("Rule I - CURRENT BEST", rule_i)

# ---------------------------------------------------------
# Experiment A
# Strong name + missing address
# ---------------------------------------------------------

rule_a = (
    rule_i
    |
    (
        (~both_address)
        & (df["name_token_similarity"] >= 0.95)
        & (df["name_similarity"] >= 0.80)
    )
)

evaluate(
    "A: missing addr + token>=.95 + name>=.80",
    rule_a
)

# ---------------------------------------------------------
# Experiment B
# Slightly stricter strong-name rule
# ---------------------------------------------------------

rule_b2 = (
    rule_i
    |
    (
        (~both_address)
        & (df["name_token_similarity"] >= 0.95)
        & (df["name_similarity"] >= 0.85)
    )
)

evaluate(
    "B: missing addr + token>=.95 + name>=.85",
    rule_b2
)

# ---------------------------------------------------------
# Experiment C
# Very strong exact-ish name
# ---------------------------------------------------------

rule_c = (
    rule_i
    |
    (
        (~both_address)
        & (df["name_token_similarity"] >= 0.98)
        & (df["name_similarity"] >= 0.85)
    )
)

evaluate(
    "C: missing addr + token>=.98 + name>=.85",
    rule_c
)

# ---------------------------------------------------------
# Experiment D
# Exact numbers + moderate address
# ---------------------------------------------------------

rule_d = (
    rule_i
    |
    (
        both_address
        & (df["number_overlap"] >= 1.0)
        & (df["address_similarity"] >= 0.70)
        & (df["address_token_similarity"] >= 0.85)
    )
)

evaluate(
    "D: exact numbers + addr>=.70 + token>=.85",
    rule_d
)

# ---------------------------------------------------------
# Experiment E
# Exact numbers + stronger address
# ---------------------------------------------------------

rule_e = (
    rule_i
    |
    (
        both_address
        & (df["number_overlap"] >= 1.0)
        & (df["address_similarity"] >= 0.75)
        & (df["address_token_similarity"] >= 0.85)
    )
)

evaluate(
    "E: exact numbers + addr>=.75 + token>=.85",
    rule_e
)

# ---------------------------------------------------------
# Experiment F
# Exact numbers + address token very strong
# ---------------------------------------------------------

rule_f = (
    rule_i
    |
    (
        both_address
        & (df["number_overlap"] >= 1.0)
        & (df["address_token_similarity"] >= 0.90)
        & (df["address_similarity"] >= 0.70)
    )
)

evaluate(
    "F: exact numbers + addr>=.70 + token>=.90",
    rule_f
)

# ---------------------------------------------------------
# Experiment G
# Exact numbers + moderate address + some name evidence
# ---------------------------------------------------------

rule_g = (
    rule_i
    |
    (
        both_address
        & (df["number_overlap"] >= 1.0)
        & (df["address_similarity"] >= 0.70)
        & (df["address_token_similarity"] >= 0.85)
        & (df["name_token_similarity"] >= 0.30)
    )
)

evaluate(
    "G: exact numbers + address + name>=.30",
    rule_g
)

# ---------------------------------------------------------
# Experiment H
# Exact numbers + stronger name
# ---------------------------------------------------------

rule_h = (
    rule_i
    |
    (
        both_address
        & (df["number_overlap"] >= 1.0)
        & (df["address_similarity"] >= 0.65)
        & (df["address_token_similarity"] >= 0.85)
        & (df["name_token_similarity"] >= 0.50)
    )
)

evaluate(
    "H: exact numbers + address + name>=.50",
    rule_h
)

print("\n" + "=" * 115)
print("DONE")
print("=" * 115)