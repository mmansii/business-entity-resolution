import pandas as pd

# Load scored data
data = pd.read_csv("scored_data.csv")

# We will test Score 3 because it had the best separation
score_column = "score_3"

print("\nThreshold Results")
print("-" * 70)

# Test different thresholds
thresholds = [
    0.40, 0.45, 0.50, 0.55, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90
]

for threshold in thresholds:

    # Predict match when score >= threshold
    data["prediction"] = (
        data[score_column] >= threshold
    ).astype(int)

    # True positives
    tp = ((data["prediction"] == 1) & (data["label"] == 1)).sum()

    # False positives
    fp = ((data["prediction"] == 1) & (data["label"] == 0)).sum()

    # False negatives
    fn = ((data["prediction"] == 0) & (data["label"] == 1)).sum()

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
        f"TP: {tp} | FP: {fp} | FN: {fn}"
    )