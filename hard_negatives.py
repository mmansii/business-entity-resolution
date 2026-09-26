import pandas as pd
from rapid_features import calculate_features


# Load data
source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=100
)

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=1000
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    nrows=100
)


# Create lookup for real matches
real_matches = {}

for _, row in ground_truth.iterrows():

    source1_id = row["source1_entity_id"]

    matched_ids = str(row["matched_entity_ids"]).split(",")

    real_matches[source1_id] = set(
        x.strip() for x in matched_ids
    )


results = []


# Compare Source 1 with Source 2
for _, row1 in source1.iterrows():

    source1_id = row1["entity_id"]

    # Get real matches for this Source 1 entity
    known_matches = real_matches.get(source1_id, set())

    for _, row2 in source2.iterrows():

        source2_id = row2["entity_id"]

        # Skip if this is a known genuine match
        if source2_id in known_matches:
            continue

        # Only compare same country
        country1 = str(row1["country"]).strip().lower()
        country2 = str(row2["country"]).strip().lower()

        if country1 != country2:
            continue

        # Calculate NEW features
        features = calculate_features(row1, row2)

        # Keep potentially confusing non-matches
        if (
            features["name_similarity"] >= 0.50
            or
            features["address_similarity"] >= 0.50
        ):

            results.append({
                "source1_id": source1_id,
                "source2_id": source2_id,
                **features,
                "label": 0
            })


# Convert to DataFrame
results_df = pd.DataFrame(results)


print("\nHARD NEGATIVES")
print("=" * 70)

print("Number of hard negatives:", len(results_df))

print("\nFirst 10 examples:")
print(
    results_df.head(10).to_string(index=False)
)


# Save
results_df.to_csv(
    "hard_negatives.csv",
    index=False
)

print("\nSaved as: hard_negatives.csv")