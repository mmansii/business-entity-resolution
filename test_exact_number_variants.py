import pandas as pd

df = pd.read_csv("matching_features_v2_labeled_numbers.csv")

# =========================================================
# COMMON FEATURES
# =========================================================

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

# =========================================================
# RULE I
# =========================================================

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

# =========================================================
# CURRENT RULE F
# =========================================================

rule_f = (
    rule_i
    |
    (
        both_address
        & (df["number_overlap"] >= 1.0)
        & (df["address_similarity"] >= 0.70)
        & (df["address_token_similarity"] >= 0.90)
    )
)

# =========================================================
# EVALUATION
# =========================================================

def evaluate(name, prediction):

    tp = ((prediction) & (df["label"] == 1)).sum()
    fp = ((prediction) & (df["label"] == 0)).sum()
    fn = ((~prediction) & (df["label"] == 1)).sum()

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0
    )

    f05 = (
        1.25 * precision * recall
        / (0.25 * precision + recall)
        if precision + recall
        else 0
    )

    print(
        f"{name:<55}"
        f" TP={tp:<5}"
        f" FP={fp:<5}"
        f" FN={fn:<5}"
        f" P={precision:.4f}"
        f" R={recall:.4f}"
        f" F0.5={f05:.4f}"
    )


print("\n" + "=" * 125)
print("EXACT NUMBER RULE VARIATIONS")
print("=" * 125)

evaluate(
    "CURRENT RULE F",
    rule_f
)

# ---------------------------------------------------------
# Variant 1
# ---------------------------------------------------------

rule_1 = (
    rule_i
    |
    (
        both_address
        & (df["number_overlap"] >= 1.0)
        & (df["address_similarity"] >= 0.65)
        & (df["address_token_similarity"] >= 0.85)
    )
)

evaluate(
    "1: exact numbers + addr>=.65 + token>=.85",
    rule_1
)

# ---------------------------------------------------------
# Variant 2
# ---------------------------------------------------------

rule_2 = (
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
    "2: exact numbers + addr>=.70 + token>=.85",
    rule_2
)

# ---------------------------------------------------------
# Variant 3
# ---------------------------------------------------------

rule_3 = (
    rule_i
    |
    (
        both_address
        & (df["number_overlap"] >= 1.0)
        & (df["address_similarity"] >= 0.60)
        & (df["address_token_similarity"] >= 0.90)
    )
)

evaluate(
    "3: exact numbers + addr>=.60 + token>=.90",
    rule_3
)

# ---------------------------------------------------------
# Variant 4
# ---------------------------------------------------------

rule_4 = (
    rule_i
    |
    (
        both_address
        & (df["number_overlap"] >= 1.0)
        & (df["address_similarity"] >= 0.65)
        & (df["address_token_similarity"] >= 0.90)
    )
)

evaluate(
    "4: exact numbers + addr>=.65 + token>=.90",
    rule_4
)

# ---------------------------------------------------------
# Variant 5
# ---------------------------------------------------------

rule_5 = (
    rule_i
    |
    (
        both_address
        & (df["number_overlap"] >= 1.0)
        & (
            (
                (df["address_similarity"] >= 0.70)
                & (df["address_token_similarity"] >= 0.85)
            )
            |
            (
                (df["address_similarity"] >= 0.65)
                & (df["address_token_similarity"] >= 0.90)
            )
        )
    )
)

evaluate(
    "5: combined .70/.85 OR .65/.90",
    rule_5
)

print("\n" + "=" * 125)
print("DONE")
print("=" * 125)