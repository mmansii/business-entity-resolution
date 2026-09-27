import pandas as pd

# Load the combined training data
data = pd.read_csv("comparison_data.csv")

# Score 1: balanced name + address
data["score_1"] = (
    0.30 * data["name_similarity"]
    + 0.20 * data["name_token_similarity"]
    + 0.30 * data["address_similarity"]
    + 0.20 * data["address_token_similarity"]
)

# Score 2: give more importance to token similarity
data["score_2"] = (
    0.40 * data["name_token_similarity"]
    + 0.40 * data["address_token_similarity"]
    + 0.20 * data["name_similarity"]
)

# Score 3: only token-based name + address
data["score_3"] = (
    0.50 * data["name_token_similarity"]
    + 0.50 * data["address_token_similarity"]
)

# Compare average scores for genuine matches and non-matches
print("\nAverage scores:")
print(
    data.groupby("label")[["score_1", "score_2", "score_3"]].mean()
)

print("\nScore ranges:")
print(
    data.groupby("label")[["score_1", "score_2", "score_3"]].agg(
        ["min", "max"]
    )
)

# Save the results
data.to_csv("scored_data.csv", index=False)

print("\nSaved scored data to scored_data.csv")