import pandas as pd


print("=" * 70)
print("TESTING RULE 7 - STRONG NAME EXCEPTION")
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
# BASE SCORE
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


# ============================================================
# FUNCTION
# ============================================================

def evaluate(prediction):

    tp = (
        (df["label"] == 1)
        & prediction
    ).sum()

    fp = (
        (df["label"] == 0)
        & prediction
    ).sum()

    fn = (
        (df["label"] == 1)
        & (~prediction)
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

    return tp, fp, fn, precision, recall, f05


# ============================================================
# BASELINE
# ============================================================

prediction = score >= 0.80

result = evaluate(prediction)

print()
print(
    "BASELINE"
)

print(
    f"TP={result[0]} "
    f"FP={result[1]} "
    f"FN={result[2]} "
    f"P={result[3]:.4f} "
    f"R={result[4]:.4f} "
    f"F0.5={result[5]:.4f}"
)


# ============================================================
# RULE 7A
# ============================================================

prediction = score >= 0.80

special = (
    both_address
    &
    (name >= 0.85)
    &
    (name_token >= 0.95)
    &
    (address_token >= 0.55)
    &
    (score >= 0.72)
)

prediction = prediction | special

result = evaluate(prediction)

print()
print(
    "RULE 7A:"
)

print(
    "name >= .85 AND "
    "name_token >= .95 AND "
    "address_token >= .55"
)

print(
    f"TP={result[0]} "
    f"FP={result[1]} "
    f"FN={result[2]} "
    f"P={result[3]:.4f} "
    f"R={result[4]:.4f} "
    f"F0.5={result[5]:.4f}"
)


# ============================================================
# RULE 7B
# ============================================================

prediction = score >= 0.80

special = (
    both_address
    &
    (name >= 0.90)
    &
    (name_token >= 0.95)
    &
    (address_token >= 0.55)
    &
    (score >= 0.72)
)

prediction = prediction | special

result = evaluate(prediction)

print()
print(
    "RULE 7B:"
)

print(
    "name >= .90 AND "
    "name_token >= .95 AND "
    "address_token >= .55"
)

print(
    f"TP={result[0]} "
    f"FP={result[1]} "
    f"FN={result[2]} "
    f"P={result[3]:.4f} "
    f"R={result[4]:.4f} "
    f"F0.5={result[5]:.4f}"
)


# ============================================================
# RULE 7C
# ============================================================

prediction = score >= 0.80

special = (
    both_address
    &
    (name >= 0.90)
    &
    (name_token >= 0.98)
    &
    (address_token >= 0.55)
    &
    (score >= 0.72)
)

prediction = prediction | special

result = evaluate(prediction)

print()
print(
    "RULE 7C:"
)

print(
    "name >= .90 AND "
    "name_token >= .98 AND "
    "address_token >= .55"
)

print(
    f"TP={result[0]} "
    f"FP={result[1]} "
    f"FN={result[2]} "
    f"P={result[3]:.4f} "
    f"R={result[4]:.4f} "
    f"F0.5={result[5]:.4f}"
)


# ============================================================
# RULE 7D
# ============================================================

prediction = score >= 0.80

special = (
    both_address
    &
    (name >= 0.95)
    &
    (name_token >= 0.98)
    &
    (address_token >= 0.55)
    &
    (score >= 0.72)
)

prediction = prediction | special

result = evaluate(prediction)

print()
print(
    "RULE 7D:"
)

print(
    "name >= .95 AND "
    "name_token >= .98 AND "
    "address_token >= .55"
)

print(
    f"TP={result[0]} "
    f"FP={result[1]} "
    f"FN={result[2]} "
    f"P={result[3]:.4f} "
    f"R={result[4]:.4f} "
    f"F0.5={result[5]:.4f}"
)


# ============================================================
# RULE 7E
# ============================================================

prediction = score >= 0.80

special = (
    both_address
    &
    (name >= 0.90)
    &
    (name_token >= 0.95)
    &
    (address_token >= 0.60)
    &
    (score >= 0.72)
)

prediction = prediction | special

result = evaluate(prediction)

print()
print(
    "RULE 7E:"
)

print(
    "name >= .90 AND "
    "name_token >= .95 AND "
    "address_token >= .60"
)

print(
    f"TP={result[0]} "
    f"FP={result[1]} "
    f"FN={result[2]} "
    f"P={result[3]:.4f} "
    f"R={result[4]:.4f} "
    f"F0.5={result[5]:.4f}"
)


print()
print("=" * 70)
print("DONE")
print("=" * 70)