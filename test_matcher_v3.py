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
# 2. Current baseline score
# ---------------------------------------------------------

def calculate_score(row):

    name_score = row["name_token_similarity"]
    address_score = row["address_token_similarity"]

    address1_present = row["address1_present"]
    address2_present = row["address2_present"]

    if address1_present == 1 and address2_present == 1:
        return 0.50 * name_score + 0.50 * address_score

    return 0.80 * name_score


data["match_score"] = data.apply(
    calculate_score,
    axis=1
)


# ---------------------------------------------------------
# 3. Address override
# ---------------------------------------------------------

def final_decision(row):

    score = row["match_score"]

    name_score = row["name_token_similarity"]
    address_score = row["address_token_similarity"]

    address_similarity = row["address_similarity"]

    address1_present = row["address1_present"]
    address2_present = row["address2_present"]

    # Normal decision
    if score >= 0.80:
        return 1

    # -----------------------------------------------------
    # Special case:
    # Very strong address similarity
    # -----------------------------------------------------

    if (
        address1_present == 1
        and address2_present == 1
        and address_score >= 0.90
        and address_similarity >= 0.70
    ):
        return 1

    return 0


data["prediction"] = data.apply(
    final_decision,
    axis=1
)


# ---------------------------------------------------------
# 4. Calculate metrics
# ---------------------------------------------------------

actual = data["label"] == 1
predicted = data["prediction"] == 1

tp = (predicted & actual).sum()
fp = (predicted & ~actual).sum()
fn = (~predicted & actual).sum()

precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0

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


# ---------------------------------------------------------
# 5. Results
# ---------------------------------------------------------

print()
print("=" * 70)
print("ADDRESS OVERRIDE RESULTS")
print("=" * 70)

print(f"TP       : {tp}")
print(f"FP       : {fp}")
print(f"FN       : {fn}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F0.5     : {f05:.3f}")


# ---------------------------------------------------------
# 6. Compare with baseline
# ---------------------------------------------------------

print()
print("=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

print("Previous baseline:")
print("Precision = 0.989")
print("Recall    = 0.817")
print("F0.5      = 0.949")

print()
print("New address-override model:")
print(f"Precision = {precision:.3f}")
print(f"Recall    = {recall:.3f}")
print(f"F0.5      = {f05:.3f}")