import pandas as pd

df = pd.read_csv("matching_features_v2_labeled_numbers.csv")

# =========================================================
# BASIC FEATURES
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
# CURRENT RULE F
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

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f05 = (
        1.25 * precision * recall
        / (0.25 * precision + recall)
        if precision + recall > 0
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
print("SOURCE-SPECIFIC MATCHING EXPERIMENTS")
print("=" * 125)

evaluate(
    "CURRENT RULE F",
    rule_f
)

# =========================================================
# S3 ONLY — slightly more tolerant name threshold
# =========================================================

s3 = df["source"] == "S3"

rule_s3_a = (
    rule_f
    |
    (
        s3
        & (base_score >= 0.75)
        & (df["name_token_similarity"] >= 0.90)
    )
)

evaluate(
    "A: S3 name-token>=.90 + score>=.75",
    rule_s3_a
)

# =========================================================
# S3 ONLY — name + address
# =========================================================

rule_s3_b = (
    rule_f
    |
    (
        s3
        & both_address
        & (df["name_token_similarity"] >= 0.80)
        & (df["address_token_similarity"] >= 0.80)
        & (base_score >= 0.70)
    )
)

evaluate(
    "B: S3 name>=.80 + addr-token>=.80 + score>=.70",
    rule_s3_b
)

# =========================================================
# S3 ONLY — strong address
# =========================================================

rule_s3_c = (
    rule_f
    |
    (
        s3
        & both_address
        & (df["address_similarity"] >= 0.85)
        & (df["address_token_similarity"] >= 0.85)
        & (df["name_token_similarity"] >= 0.30)
    )
)

evaluate(
    "C: S3 address>=.85 + token>=.85 + name>=.30",
    rule_s3_c
)

# =========================================================
# S3 ONLY — exact numbers + lower address threshold
# =========================================================

rule_s3_d = (
    rule_f
    |
    (
        s3
        & both_address
        & (df["number_overlap"] >= 1.0)
        & (df["address_similarity"] >= 0.60)
        & (df["address_token_similarity"] >= 0.85)
    )
)

evaluate(
    "D: S3 exact numbers + addr>=.60 + token>=.85",
    rule_s3_d
)

# =========================================================
# S2 ONLY — same tests for comparison
# =========================================================

s2 = df["source"] == "S2"

rule_s2_a = (
    rule_f
    |
    (
        s2
        & (base_score >= 0.75)
        & (df["name_token_similarity"] >= 0.90)
    )
)

evaluate(
    "E: S2 name-token>=.90 + score>=.75",
    rule_s2_a
)

rule_s2_b = (
    rule_f
    |
    (
        s2
        & both_address
        & (df["name_token_similarity"] >= 0.80)
        & (df["address_token_similarity"] >= 0.80)
        & (base_score >= 0.70)
    )
)

evaluate(
    "F: S2 name>=.80 + addr-token>=.80 + score>=.70",
    rule_s2_b
)

rule_s2_c = (
    rule_f
    |
    (
        s2
        & both_address
        & (df["address_similarity"] >= 0.85)
        & (df["address_token_similarity"] >= 0.85)
        & (df["name_token_similarity"] >= 0.30)
    )
)

evaluate(
    "G: S2 address>=.85 + token>=.85 + name>=.30",
    rule_s2_c
)

rule_s2_d = (
    rule_f
    |
    (
        s2
        & both_address
        & (df["number_overlap"] >= 1.0)
        & (df["address_similarity"] >= 0.60)
        & (df["address_token_similarity"] >= 0.85)
    )
)

evaluate(
    "H: S2 exact numbers + addr>=.60 + token>=.85",
    rule_s2_d
)

print("\n" + "=" * 125)
print("DONE")
print("=" * 125)