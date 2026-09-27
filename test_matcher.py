import pandas as pd


# =========================================================
# 1. LOAD VALIDATION DATA
# =========================================================

print("\nLoading validation data...")

data = pd.read_csv("validation_data.csv")

print(f"Total validation pairs: {len(data)}")

print(
    f"Positive pairs: {(data['label'] == 1).sum()}"
)

print(
    f"Negative pairs: {(data['label'] == 0).sum()}"
)


# =========================================================
# 2. CREATE MATCHING SCORE
# =========================================================
#
# When BOTH addresses exist:
#
#   50% name
#   50% address
#
# When an address is missing:
#
#   use name similarity
#
# But we don't trust name similarity completely.
#
# So missing-address matches receive a penalty.
#
# =========================================================


def calculate_score(row):

    name_score = row["name_token_similarity"]

    address_score = row["address_token_similarity"]

    address1_present = row["address1_present"]

    address2_present = row["address2_present"]


    # -----------------------------------------------------
    # BOTH addresses exist
    # -----------------------------------------------------

    if address1_present == 1 and address2_present == 1:

        score = (
            0.50 * name_score
            +
            0.50 * address_score
        )

        return score


    # -----------------------------------------------------
    # One or both addresses are missing
    # -----------------------------------------------------
    #
    # We use name similarity but reduce confidence.
    #
    # -----------------------------------------------------

    score = 0.80 * name_score

    return score


data["match_score"] = data.apply(
    calculate_score,
    axis=1
)


# =========================================================
# 3. DISPLAY SCORE STATISTICS
# =========================================================

print("\nScore statistics")
print("=" * 70)

print("\nNegative pairs:")

print(
    data[data["label"] == 0]["match_score"].describe()
)

print("\nPositive pairs:")

print(
    data[data["label"] == 1]["match_score"].describe()
)


# =========================================================
# 4. CALCULATE METRICS
# =========================================================


def calculate_metrics(data, threshold):

    predicted = (
        data["match_score"] >= threshold
    )

    actual = (
        data["label"] == 1
    )


    tp = (
        predicted & actual
    ).sum()

    fp = (
        predicted & ~actual
    ).sum()

    fn = (
        ~predicted & actual
    ).sum()


    if tp + fp == 0:

        precision = 0

    else:

        precision = tp / (tp + fp)


    if tp + fn == 0:

        recall = 0

    else:

        recall = tp / (tp + fn)


    # F0.5 gives more importance to precision.
    beta = 0.5

    if precision == 0 and recall == 0:

        f05 = 0

    else:

        f05 = (
            (1 + beta**2)
            * precision
            * recall
            /
            (
                beta**2 * precision
                +
                recall
            )
        )


    return tp, fp, fn, precision, recall, f05


# =========================================================
# 5. TEST DIFFERENT THRESHOLDS
# =========================================================

print("\n")
print("=" * 90)
print("THRESHOLD TEST")
print("=" * 90)

print(
    f"{'Threshold':<12}"
    f"{'TP':<8}"
    f"{'FP':<8}"
    f"{'FN':<8}"
    f"{'Precision':<12}"
    f"{'Recall':<10}"
    f"{'F0.5':<10}"
)

print("-" * 90)


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

    tp, fp, fn, precision, recall, f05 = (
        calculate_metrics(
            data,
            threshold
        )
    )


    print(
        f"{threshold:<12.2f}"
        f"{tp:<8}"
        f"{fp:<8}"
        f"{fn:<8}"
        f"{precision:<12.3f}"
        f"{recall:<10.3f}"
        f"{f05:<10.3f}"
    )


print("\nMatcher test completed.")