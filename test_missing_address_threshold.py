import pandas as pd

print("Loading validation data...")
data = pd.read_csv("validation_data.csv")


# ---------------------------------------------------------
# BASELINE SCORE
# ---------------------------------------------------------

def calculate_score(row):

    name_score = row["name_token_similarity"]
    address_score = row["address_token_similarity"]

    # Both addresses available
    if row["address1_present"] == 1 and row["address2_present"] == 1:
        return 0.50 * name_score + 0.50 * address_score

    # One or both addresses missing
    return 0.80 * name_score


data["match_score"] = data.apply(calculate_score, axis=1)


# ---------------------------------------------------------
# RULE J + MISSING ADDRESS THRESHOLD
# ---------------------------------------------------------

def predict(row, missing_name_threshold):

    # Normal Rule J score
    if row["match_score"] >= 0.80:
        return 1

    # Address override from Rule J
    if (
        row["address1_present"] == 1
        and row["address2_present"] == 1
        and row["address_token_similarity"] >= 0.70
        and row["address_similarity"] >= 0.85
    ):
        return 1

    return 0


# ---------------------------------------------------------
# TEST DIFFERENT NAME THRESHOLDS
# ---------------------------------------------------------

thresholds = [0.80, 0.85, 0.90, 0.95, 1.00]

print()
print("=" * 90)
print("TESTING MISSING-ADDRESS NAME THRESHOLDS")
print("=" * 90)

for threshold in thresholds:

    def predict_test(row):

        # Both addresses available
        if (
            row["address1_present"] == 1
            and row["address2_present"] == 1
        ):

            # Normal score
            if row["match_score"] >= 0.80:
                return 1

            # Address override
            if (
                row["address_token_similarity"] >= 0.70
                and row["address_similarity"] >= 0.85
            ):
                return 1

            return 0

        # Missing address:
        # require stronger name similarity
        return int(row["name_token_similarity"] >= threshold)


    data["prediction"] = data.apply(predict_test, axis=1)

    tp = ((data["label"] == 1) & (data["prediction"] == 1)).sum()
    fp = ((data["label"] == 0) & (data["prediction"] == 1)).sum()
    fn = ((data["label"] == 1) & (data["prediction"] == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    beta_squared = 0.25

    f05 = (
        (1 + beta_squared)
        * precision
        * recall
        / (beta_squared * precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"TP={tp} | "
        f"FP={fp} | "
        f"FN={fn} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f} | "
        f"F0.5={f05:.3f}"
    )