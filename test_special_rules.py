import pandas as pd


print("=" * 70)
print("TESTING TARGETED MATCHING EXCEPTIONS")
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
# EVALUATION
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

    return (
        tp,
        fp,
        fn,
        precision,
        recall,
        f05
    )


# ============================================================
# BASELINE
# ============================================================

base_prediction = (
    score >= 0.80
).astype(int)


result = evaluate(
    base_prediction
)


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
# SPECIAL RULE 1
# ============================================================

prediction = base_prediction.copy()


special = (
    both_address
    &
    (name_token >= 0.95)
    &
    (address_token >= 0.60)
    &
    (address >= 0.40)
    &
    (score >= 0.70)
)


prediction.loc[special] = 1


result = evaluate(
    prediction
)


print()
print(
    "RULE 1: strong name + reasonable address"
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
# SPECIAL RULE 2
# ============================================================

prediction = base_prediction.copy()


special = (
    both_address
    &
    (name_token >= 0.90)
    &
    (address_token >= 0.70)
    &
    (score >= 0.70)
)


prediction.loc[special] = 1


result = evaluate(
    prediction
)


print()
print(
    "RULE 2: strong name + strong token address"
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
# SPECIAL RULE 3
# ============================================================

prediction = base_prediction.copy()


special = (
    both_address
    &
    (name_token >= 0.80)
    &
    (address_token >= 0.85)
    &
    (score >= 0.70)
)


prediction.loc[special] = 1


result = evaluate(
    prediction
)


print()
print(
    "RULE 3: good name + very strong address"
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
# SPECIAL RULE 4
# ============================================================

prediction = base_prediction.copy()


special = (
    both_address
    &
    (name_token >= 0.70)
    &
    (address_token >= 0.90)
    &
    (score >= 0.70)
)


prediction.loc[special] = 1


result = evaluate(
    prediction
)


print()
print(
    "RULE 4: moderate name + extremely strong address"
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
# SPECIAL RULE 5
# ============================================================

prediction = base_prediction.copy()


special = (
    both_address
    &
    (name_token >= 0.85)
    &
    (address_token >= 0.80)
    &
    (address >= 0.50)
    &
    (score >= 0.70)
)


prediction.loc[special] = 1


result = evaluate(
    prediction
)


print()
print(
    "RULE 5: strong name + solid address"
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
# SPECIAL RULE 6
# ============================================================

prediction = base_prediction.copy()


special = (
    both_address
    &
    (name_token >= 0.95)
    &
    (address_token >= 0.55)
    &
    (score >= 0.72)
)


prediction.loc[special] = 1


result = evaluate(
    prediction
)


print()
print(
    "RULE 6: very strong name + moderate address"
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