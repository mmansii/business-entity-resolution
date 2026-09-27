import pandas as pd

# Load scored data
data = pd.read_csv("scored_data.csv")


def missing_aware_score(row):
    name = row["name_token_similarity"]
    address = row["address_token_similarity"]

    name_present = row["address1_present"] == 1
    address_present = row["address2_present"] == 1

    # Both records have an address
    if name_present and address_present:
        return 0.5 * name + 0.5 * address

    # At least one address is missing
    # In that case, rely on the name
    return name


# Calculate new score
data["missing_aware_score"] = data.apply(
    missing_aware_score,
    axis=1
)


print("\nMissing-Aware Score")
print("=" * 70)

print(
    data.groupby("label")["missing_aware_score"].mean()
)

print("\nScore ranges")
print(
    data.groupby("label")["missing_aware_score"].agg(
        ["min", "max"]
    )
)


# Test thresholds
print("\nThreshold Results")
print("-" * 90)

thresholds = [
    0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85,
    0.90, 0.95
]


for threshold in thresholds:

    data["prediction"] = (
        data["missing_aware_score"] >= threshold
    ).astype(int)

    tp = (
        (data["prediction"] == 1) &
        (data["label"] == 1)
    ).sum()

    fp = (
        (data["prediction"] == 1) &
        (data["label"] == 0)
    ).sum()

    fn = (
        (data["prediction"] == 0) &
        (data["label"] == 1)
    ).sum()

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

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


# Save results
data.to_csv(
    "missing_aware_scores.csv",
    index=False
)

print("\nSaved to missing_aware_scores.csv")