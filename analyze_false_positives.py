import pandas as pd


print("Loading validation data...")

data = pd.read_csv("validation_data.csv")


# ---------------------------------------------------------
# 1. Baseline score
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
# 2. Rule J
# ---------------------------------------------------------

def predict(row):

    # Normal matcher
    if row["match_score"] >= 0.80:
        return 1

    # Address override - Rule J
    if (
        row["address1_present"] == 1
        and row["address2_present"] == 1
        and row["address_token_similarity"] >= 0.70
        and row["address_similarity"] >= 0.85
    ):
        return 1

    return 0


data["prediction"] = data.apply(
    predict,
    axis=1
)


# ---------------------------------------------------------
# 3. Find false positives
# ---------------------------------------------------------

false_positives = data[
    (data["label"] == 0) &
    (data["prediction"] == 1)
].copy()


print()
print("=" * 70)
print("FALSE POSITIVE ANALYSIS - RULE J")
print("=" * 70)

print(
    f"False positives: {len(false_positives)}"
)


# ---------------------------------------------------------
# 4. Display them
# ---------------------------------------------------------

columns = [
    "source1_id",
    "matched_id",
    "source",
    "name_similarity",
    "name_token_similarity",
    "address_similarity",
    "address_token_similarity",
    "address1_present",
    "address2_present",
    "match_score"
]


print()
print(false_positives[columns].to_string(index=False))


# ---------------------------------------------------------
# 5. Show why each was accepted
# ---------------------------------------------------------

print()
print("=" * 70)
print("WHY WERE THEY ACCEPTED?")
print("=" * 70)

for _, row in false_positives.iterrows():

    print()
    print(f"S1: {row['source1_id']}")
    print(f"Matched ID: {row['matched_id']}")
    print(f"Source: {row['source']}")

    print(
        f"Name token similarity: "
        f"{row['name_token_similarity']:.3f}"
    )

    print(
        f"Address token similarity: "
        f"{row['address_token_similarity']:.3f}"
    )

    print(
        f"Address similarity: "
        f"{row['address_similarity']:.3f}"
    )

    print(
        f"Normal score: "
        f"{row['match_score']:.3f}"
    )

    if row["match_score"] >= 0.80:
        print("Accepted because: NORMAL SCORE >= 0.80")
    else:
        print("Accepted because: ADDRESS OVERRIDE")


# ---------------------------------------------------------
# 6. Print ALL false positives clearly
# ---------------------------------------------------------

print()
print("=" * 70)
print("ALL FALSE POSITIVE IDs")
print("=" * 70)

for i, (_, row) in enumerate(false_positives.iterrows(), start=1):

    print(
        f"{i}. "
        f"{row['source1_id']} -> "
        f"{row['matched_id']} "
        f"| source={row['source']} "
        f"| score={row['match_score']:.6f} "
        f"| name={row['name_token_similarity']:.6f} "
        f"| address_token={row['address_token_similarity']:.6f} "
        f"| address={row['address_similarity']:.6f}"
    )


# ---------------------------------------------------------
# 7. Save
# ---------------------------------------------------------

false_positives.to_csv(
    "false_positives_rule_j.csv",
    index=False
)

print()
print("=" * 70)
print("Saved to:")
print("false_positives_rule_j.csv")
print("=" * 70)