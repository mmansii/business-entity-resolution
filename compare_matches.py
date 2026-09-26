import pandas as pd

# Load genuine matches
positive = pd.read_csv("real_positive_features.csv")

# Load hard non-matches
negative = pd.read_csv("hard_negatives.csv")


# Add labels
positive["label"] = 1
negative["label"] = 0


# Combine them
data = pd.concat(
    [positive, negative],
    ignore_index=True
)


print("\nDATASET SUMMARY")
print("=" * 70)

print("Total examples:", len(data))
print("Genuine matches:", len(positive))
print("Non-matches:", len(negative))


# Features we want to compare
features = [
    "name_similarity",
    "name_token_similarity",
    "address_similarity",
    "address_token_similarity",
    "country_match"
]


print("\nAVERAGE FEATURES")
print("=" * 70)

print(
    data.groupby("label")[features].mean()
)


print("\nMINIMUM / MAXIMUM FOR GENUINE MATCHES")
print("=" * 70)

print(
    positive[features].agg(["min", "max", "mean"])
)


print("\nMINIMUM / MAXIMUM FOR HARD NON-MATCHES")
print("=" * 70)

print(
    negative[features].agg(["min", "max", "mean"])
)


# Save combined data
data.to_csv("comparison_data.csv", index=False)

print("\nSaved as: comparison_data.csv")