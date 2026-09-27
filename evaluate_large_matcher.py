import pandas as pd
import numpy as np


print("=" * 70)
print("EVALUATING MATCHER ON LARGE BLOCKED VALIDATION SET")
print("=" * 70)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading matching features...")

features = pd.read_csv(
    "matching_features_v2.csv"
)

print(
    "Feature rows:",
    len(features)
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print(
    "Ground truth rows:",
    len(ground_truth)
)


# ============================================================
# KEEP ONLY THE FIRST 1000 SOURCE 1 ENTITIES
# ============================================================

source1_ids = set(
    features["source1_id"].unique()
)

print(
    "Source 1 entities in validation:",
    len(source1_ids)
)


ground_truth = ground_truth[
    ground_truth["source1_entity_id"].isin(
        source1_ids
    )
].copy()


# ============================================================
# CREATE SET OF TRUE MATCHES
# ============================================================

print("\nCreating ground-truth match set...")


true_matches = set()


for _, row in ground_truth.iterrows():

    source1_id = row["source1_entity_id"]

    matched_ids = str(
        row["matched_entity_ids"]
    )

    if matched_ids == "nan":
        continue

    if matched_ids.strip() == "":
        continue


    for matched_id in matched_ids.split(","):

        matched_id = matched_id.strip()

        if matched_id:

            true_matches.add(
                (
                    source1_id,
                    matched_id
                )
            )


print(
    "True matches:",
    len(true_matches)
)


# ============================================================
# ADD LABEL
# ============================================================

print("\nAdding labels...")


features["label"] = [
    int(
        (row.source1_id, row.matched_id)
        in true_matches
    )
    for row in features.itertuples()
]


print(
    "Positive candidates:",
    features["label"].sum()
)

print(
    "Negative candidates:",
    len(features)
    - features["label"].sum()
)


# ============================================================
# RULE J
# ============================================================

def calculate_rule_j(row):

    name_score = row.name_token_similarity

    address_score = row.address_token_similarity

    address_similarity = row.address_similarity

    address1_present = row.address1_present

    address2_present = row.address2_present


    # --------------------------------------------------------
    # NORMAL SCORE
    # --------------------------------------------------------

    if (
        address1_present == 1
        and address2_present == 1
    ):

        normal_score = (
            0.50 * name_score
            +
            0.50 * address_score
        )

    else:

        normal_score = (
            0.80 * name_score
        )


    # --------------------------------------------------------
    # RULE J
    # --------------------------------------------------------

    if normal_score >= 0.80:

        return 1


    # Address override
    if (
        address1_present == 1
        and address2_present == 1
        and address_score >= 0.70
        and address_similarity >= 0.85
    ):

        return 1


    return 0


print("\nApplying Rule J...")


features["prediction"] = features.apply(
    calculate_rule_j,
    axis=1
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

true_positive = (
    (features["label"] == 1)
    &
    (features["prediction"] == 1)
).sum()


false_positive = (
    (features["label"] == 0)
    &
    (features["prediction"] == 1)
).sum()


false_negative = (
    (features["label"] == 1)
    &
    (features["prediction"] == 0)
).sum()


true_negative = (
    (features["label"] == 0)
    &
    (features["prediction"] == 0)
).sum()


# ============================================================
# METRICS
# ============================================================

precision = (
    true_positive
    /
    (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)


recall = (
    true_positive
    /
    (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)


f05 = (
    1.25
    * precision
    * recall
    /
    (
        0.25 * precision
        + recall
    )
    if (precision + recall) > 0
    else 0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("RULE J RESULTS")
print("=" * 70)

print()

print("True Positives :", true_positive)
print("False Positives:", false_positive)
print("False Negatives:", false_negative)
print("True Negatives :", true_negative)

print()

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F0.5:",
    round(f05, 4)
)


# ============================================================
# SAVE LABELED DATA
# ============================================================

features.to_csv(
    "matching_features_v2_labeled.csv",
    index=False
)


print()
print(
    "Saved:",
    "matching_features_v2_labeled.csv"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)