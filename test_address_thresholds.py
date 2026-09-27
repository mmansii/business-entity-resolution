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

address_token_thresholds = [
    0.85,
    0.90,
    0.95,
    1.00
]

address_similarity_thresholds = [
    0.60,
    0.70,
    0.80,
    0.90
]


print()
print("=" * 90)
print("ADDRESS THRESHOLD EXPERIMENT")
print("=" * 90)

print(
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


for token_threshold in address_token_thresholds:

    for similarity_threshold in address_similarity_thresholds:

        def predict(row):

            # Normal matcher
            if row["match_score"] >= 0.80:
                return 1

            # Address override
            if (
                row["address1_present"] == 1
                and row["address2_present"] == 1
                and row["address_token_similarity"]
                    >= token_threshold
                and row["address_similarity"]
                    >= similarity_threshold
            ):
                return 1

            return 0


        predictions = data.apply(
            predict,
            axis=1
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
            f"{token_threshold:8.2f} "
            f"{similarity_threshold:10.2f} "
            f"{tp:6d} "
            f"{fp:6d} "
            f"{fn:6d} "
            f"{precision:10.3f} "
            f"{recall:8.3f} "
            f"{f05:8.3f}"
        )