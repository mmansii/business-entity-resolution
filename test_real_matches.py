import pandas as pd
from rapid_features import calculate_features


# Read the ground truth
ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

# Read Source 1, Source 2 and Source 3
source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)


# Create quick ID lookups
source1_lookup = source1.set_index("entity_id").to_dict("index")
source2_lookup = source2.set_index("entity_id").to_dict("index")
source3_lookup = source3.set_index("entity_id").to_dict("index")


results = []

# Test first 20 Source 1 businesses
for _, gt_row in ground_truth.head(20).iterrows():

    source1_id = gt_row["source1_entity_id"]
    matched_ids = str(gt_row["matched_entity_ids"]).split(",")

    business1 = source1_lookup[source1_id]

    for matched_id in matched_ids:

        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):
            business2 = source2_lookup[matched_id]
        elif matched_id.startswith("S3-"):
            business2 = source3_lookup[matched_id]
        else:
            continue

        features = calculate_features(business1, business2)

        results.append({
            "source1_id": source1_id,
            "matched_id": matched_id,
            **features,
            "label": 1
        })


# Convert to DataFrame
results_df = pd.DataFrame(results)

print("\nREAL POSITIVE MATCHES")
print("=" * 70)

print("Number of genuine matches tested:", len(results_df))

print("\nAverage feature values:")
print(
    results_df[
        [
            "name_similarity",
            "name_token_similarity",
            "address_similarity",
            "address_token_similarity",
            "country_match"
        ]
    ].mean()
)

print("\nFirst 15 genuine matches:")
print(results_df.head(15).to_string(index=False))

# Save results
results_df.to_csv("real_positive_features.csv", index=False)

print("\nSaved as: real_positive_features.csv")