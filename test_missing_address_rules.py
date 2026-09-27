import pandas as pd

# Load validation data
data = pd.read_csv("validation_data.csv")


# ---------------------------------------------------------
# Create several alternative scoring rules
# ---------------------------------------------------------

def calculate_score(row, rule):

    name = row["name_token_similarity"]
    address = row["address_token_similarity"]

    address1_present = row["address1_present"] == 1
    address2_present = row["address2_present"] == 1

    # Rule 1:
    # Current method
    if rule == 1:

        if address1_present and address2_present:
            return 0.5 * name + 0.5 * address

        return name

    # Rule 2:
    # If one address is missing, reduce confidence by 10%
    elif rule == 2:

        if address1_present and address2_present:
            return 0.5 * name + 0.5 * address

        return 0.90 * name

    # Rule 3:
    # If one address is missing, reduce confidence by 20%
    elif rule == 3:

        if address1_present and address2_present:
            return 0.5 * name + 0.5 * address

        return 0.80 * name

    # Rule 4:
    # If one address is missing, reduce confidence by 30%
    elif rule == 4:

        if address1_present and address2_present:
            return 0.5 * name + 0.5 * address

        return 0.70 * name


# ---------------------------------------------------------
# F0.5 calculation
# ---------------------------------------------------------

def calculate_metrics(data, threshold):

    prediction = (
        data["score"] >= threshold
    ).astype(int)

    tp = (
        (prediction == 1) &
        (data["label"] == 1)
    ).sum()

    fp = (
        (prediction == 1) &
        (data["label"] == 0)
    ).sum()

    fn = (
        (prediction == 0) &
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

    return precision, recall, f05, tp, fp, fn


# ---------------------------------------------------------
# Test rules
# ---------------------------------------------------------

thresholds = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

print("\nComparing Missing-Address Rules")
print("=" * 110)

for rule in range(1, 5):

    data["score"] = data.apply(
        lambda row: calculate_score(row, rule),
        axis=1
    )

    print(f"\nRULE {rule}")
    print("-" * 110)

    if rule == 1:
        print("Both addresses: 50/50 name + address")
        print("Missing address: name only")

    elif rule == 2:
        print("Both addresses: 50/50 name + address")
        print("Missing address: name × 0.90")

    elif rule == 3:
        print("Both addresses: 50/50 name + address")
        print("Missing address: name × 0.80")

    elif rule == 4:
        print("Both addresses: 50/50 name + address")
        print("Missing address: name × 0.70")

    print()

    for threshold in thresholds:

        precision, recall, f05, tp, fp, fn = calculate_metrics(
            data,
            threshold
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
        