import pandas as pd

# Load validation dataset
data = pd.read_csv("validation_data.csv")

print("\nValidation dataset loaded")
print("=" * 70)
print(f"Total examples: {len(data)}")

# ---------------------------------------------------------
# Missing-aware scoring
# ---------------------------------------------------------
#
# If both addresses exist:
#     50% name token similarity
#     50% address token similarity
#
# If either address is missing:
#     rely mainly on name similarity
#
# This is important because a genuine business match
# can have a missing address in one source.
# ---------------------------------------------------------

def missing_aware_score(row):

    name_score = row["name_token_similarity"]
    address_score = row["address_token_similarity"]

    address1_present = row["address1_present"] == 1
    address2_present = row["address2_present"] == 1

    # Both addresses are available
    if address1_present and address2_present:
        return 0.5 * name_score + 0.5 * address_score

    # One or both addresses are missing
    return name_score


data["score"] = data.apply(missing_aware_score, axis=1)


# ---------------------------------------------------------
# Basic score comparison
# ---------------------------------------------------------

print("\nAverage score by label")
print("=" * 70)

print(
    data.groupby("label")["score"].mean()
)


print("\nScore range by label")
print("=" * 70)

print(
    data.groupby("label")["score"].agg(["min", "max"])
)


# ---------------------------------------------------------
# Test different thresholds
# ---------------------------------------------------------

print("\nThreshold Results")
print("=" * 100)

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]


for threshold in thresholds:

    # Predict match if score >= threshold
    data["prediction"] = (
        data["score"] >= threshold
    ).astype(int)

    # True positives
    tp = (
        (data["prediction"] == 1) &
        (data["label"] == 1)
    ).sum()

    # False positives
    fp = (
        (data["prediction"] == 1) &
        (data["label"] == 0)
    ).sum()

    # False negatives
    fn = (
        (data["prediction"] == 0) &
        (data["label"] == 1)
    ).sum()

    # Precision
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    # Recall
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    # F0.5
    if precision == 0 and recall == 0:
        f05 = 0
    else:
        f05 = (
            1.25 * precision * recall
            / (0.25 * precision + recall)
        )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F0.5: {f05:.3f} | "
        f"TP: {tp} | "
        f"FP: {fp} | "
        f"FN: {fn}"
    )


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

data.to_csv(
    "validation_scored.csv",
    index=False
)

print("\nSaved results to:")
print("validation_scored.csv")