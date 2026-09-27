import pandas as pd


# ---------------------------------------------------------
# 1. Load validation data
# ---------------------------------------------------------

print("Loading validation data...")

data = pd.read_csv("validation_data.csv")

print(f"Total validation pairs: {len(data)}")
print(f"Positive pairs: {data['label'].sum()}")
print(f"Negative pairs: {(data['label'] == 0).sum()}")


# ---------------------------------------------------------
# 2. New scoring function
# ---------------------------------------------------------

def calculate_score(row):

    name_score = row["name_token_similarity"]
    address_score = row["address_token_similarity"]

    address1_present = row["address1_present"]
    address2_present = row["address2_present"]

    # -----------------------------------------------------
    # Both addresses available
    # -----------------------------------------------------

    if address1_present == 1 and address2_present == 1:

        # Give more importance to address
        score = (
            0.30 * name_score +
            0.70 * address_score
        )

        return score

    # -----------------------------------------------------
    # Address missing
    # -----------------------------------------------------

    return 0.80 * name_score


data["match_score"] = data.apply(calculate_score, axis=1)


# ---------------------------------------------------------
# 3. Evaluate different thresholds
# ---------------------------------------------------------

print()
print("=" * 70)
print("NEW SCORING MODEL")
print("=" * 70)

thresholds = [
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

    predicted = data["match_score"] >= threshold

    actual = data["label"] == 1

    tp = (predicted & actual).sum()
    fp = (predicted & ~actual).sum()
    fn = (~predicted & actual).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    beta = 0.5

    f05 = (
        (1 + beta**2)
        * precision
        * recall
        /
        (
            beta**2 * precision + recall
        )
        if (precision + recall) > 0
        else 0
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"TP {tp:3d} | "
        f"FP {fp:4d} | "
        f"FN {fn:3d} | "
        f"P {precision:.3f} | "
        f"R {recall:.3f} | "
        f"F0.5 {f05:.3f}"
    )