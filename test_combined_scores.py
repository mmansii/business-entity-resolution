import pandas as pd


print("=" * 70)
print("TESTING COMBINED NAME + ADDRESS SCORES")
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
# EVALUATION FUNCTION
# ============================================================

def evaluate(prediction):

    tp = (
        (df["label"] == 1)
        & (prediction == 1)
    ).sum()

    fp = (
        (df["label"] == 0)
        & (prediction == 1)
    ).sum()

    fn = (
        (df["label"] == 1)
        & (prediction == 0)
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
# TEST COMBINED WEIGHTS
# ============================================================

print()
print("=" * 70)
print("COMBINED SCORE EXPERIMENTS")
print("=" * 70)
print()


# ------------------------------------------------------------
# Score 1
# ------------------------------------------------------------

score = (
    0.30 * name
    +
    0.20 * name_token
    +
    0.30 * address
    +
    0.20 * address_token
)

prediction = (
    score >= 0.80
).astype(int)

result = evaluate(prediction)

print(
    "30N + 20NT + 30A + 20AT"
)

print(
    f"TP={result[0]} "
    f"FP={result[1]} "
    f"FN={result[2]} "
    f"P={result[3]:.4f} "
    f"R={result[4]:.4f} "
    f"F0.5={result[5]:.4f}"
)


# ------------------------------------------------------------
# Score 2
# ------------------------------------------------------------

score = (
    0.40 * name
    +
    0.20 * name_token
    +
    0.20 * address
    +
    0.20 * address_token
)

prediction = (
    score >= 0.80
).astype(int)

result = evaluate(prediction)

print()

print(
    "40N + 20NT + 20A + 20AT"
)

print(
    f"TP={result[0]} "
    f"FP={result[1]} "
    f"FN={result[2]} "
    f"P={result[3]:.4f} "
    f"R={result[4]:.4f} "
    f"F0.5={result[5]:.4f}"
)


# ------------------------------------------------------------
# Score 3
# ------------------------------------------------------------

score = (
    0.20 * name
    +
    0.40 * name_token
    +
    0.20 * address
    +
    0.20 * address_token
)

prediction = (
    score >= 0.80
).astype(int)

result = evaluate(prediction)

print()

print(
    "20N + 40NT + 20A + 20AT"
)

print(
    f"TP={result[0]} "
    f"FP={result[1]} "
    f"FN={result[2]} "
    f"P={result[3]:.4f} "
    f"R={result[4]:.4f} "
    f"F0.5={result[5]:.4f}"
)


# ------------------------------------------------------------
# Score 4
# ------------------------------------------------------------

score = (
    0.40 * name
    +
    0.30 * name_token
    +
    0.10 * address
    +
    0.20 * address_token
)

prediction = (
    score >= 0.80
).astype(int)

result = evaluate(prediction)

print()

print(
    "40N + 30NT + 10A + 20AT"
)

print(
    f"TP={result[0]} "
    f"FP={result[1]} "
    f"FN={result[2]} "
    f"P={result[3]:.4f} "
    f"R={result[4]:.4f} "
    f"F0.5={result[5]:.4f}"
)


# ------------------------------------------------------------
# Score 5
# ------------------------------------------------------------

score = (
    0.25 * name
    +
    0.35 * name_token
    +
    0.15 * address
    +
    0.25 * address_token
)

prediction = (
    score >= 0.80
).astype(int)

result = evaluate(prediction)

print()

print(
    "25N + 35NT + 15A + 25AT"
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
# MISSING ADDRESS EXPERIMENT
# ============================================================

print()
print("=" * 70)
print("MISSING ADDRESS HANDLING")
print("=" * 70)
print()


# Combined name score
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


prediction = (
    score >= 0.80
).astype(int)


result = evaluate(prediction)


print(
    "Combined name + address token"
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
# STRICT HIGH-CONFIDENCE RULE
# ============================================================

print()
print("=" * 70)
print("HIGH-CONFIDENCE COMBINED RULE")
print("=" * 70)
print()


prediction = (
    (
        both_address
        &
        (name_token >= 0.70)
        &
        (address_token >= 0.70)
    )
    |
    (
        both_address
        &
        (name_token >= 0.50)
        &
        (address_token >= 0.90)
    )
    |
    (
        (~both_address)
        &
        (name_token >= 0.95)
    )
).astype(int)


result = evaluate(prediction)


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