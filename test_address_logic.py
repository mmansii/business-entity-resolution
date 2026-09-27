import pandas as pd


print("Loading validation data...")

data = pd.read_csv("validation_data.csv")


# ---------------------------------------------------------
# Baseline score
# ---------------------------------------------------------

def calculate_score(row):

    name_score = row["name_token_similarity"]
    address_score = row["address_token_similarity"]

    if (
        row["address1_present"] == 1
        and row["address2_present"] == 1
    ):
        return 0.50 * name_score + 0.50 * address_score

    return 0.80 * name_score


data["match_score"] = data.apply(
    calculate_score,
    axis=1
)


# ---------------------------------------------------------
# Test different address rules
# ---------------------------------------------------------

rules = [

    # Very high token similarity alone
    ("A", 0.95, 0.00),

    ("B", 0.98, 0.00),

    ("C", 1.00, 0.00),

    # High token + moderate character similarity
    ("D", 0.85, 0.60),

    ("E", 0.90, 0.60),

    ("F", 0.95, 0.60),

    # Moderate token + high character similarity
    ("G", 0.80, 0.75),

    ("H", 0.80, 0.80),

    ("I", 0.75, 0.80),

    ("J", 0.70, 0.85),
]


print()
print("=" * 90)
print("ADDRESS LOGIC EXPERIMENT")
print("=" * 90)

print(
    f"{'Rule':>6} "
    f"{'Token':>8} "
    f"{'Address':>10} "
    f"{'TP':>6} "
    f"{'FP':>6} "
    f"{'FN':>6} "
    f"{'Precision':>10} "
    f"{'Recall':>8} "
    f"{'F0.5':>8}"
)

print("-" * 90)


for rule_name, token_threshold, address_threshold in rules:

    predictions = []

    for _, row in data.iterrows():

        # Normal matcher
        if row["match_score"] >= 0.80:
            predictions.append(1)
            continue

        # Addresses must exist
        if (
            row["address1_present"] == 1
            and row["address2_present"] == 1
        ):

            token_match = (
                row["address_token_similarity"]
                >= token_threshold
            )

            # Rules A/B/C:
            # token similarity alone
            if address_threshold == 0.00:

                if token_match:
                    predictions.append(1)
                    continue

            # Other rules:
            # both conditions must be satisfied
            else:

                address_match = (
                    row["address_similarity"]
                    >= address_threshold
                )

                if token_match and address_match:
                    predictions.append(1)
                    continue

        predictions.append(0)


    predictions = pd.Series(
        predictions,
        index=data.index
    )

    actual = data["label"] == 1
    predicted = predictions == 1

    tp = (predicted & actual).sum()
    fp = (predicted & ~actual).sum()
    fn = (~predicted & actual).sum()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    beta = 0.5

    f05 = (
        (1 + beta**2)
        * precision
        * recall
        /
        (
            beta**2 * precision + recall
        )
        if precision + recall > 0
        else 0
    )

    print(
        f"{rule_name:>6} "
        f"{token_threshold:8.2f} "
        f"{address_threshold:10.2f} "
        f"{tp:6d} "
        f"{fp:6d} "
        f"{fn:6d} "
        f"{precision:10.3f} "
        f"{recall:8.3f} "
        f"{f05:8.3f}"
    )