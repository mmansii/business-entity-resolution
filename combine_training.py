import pandas as pd

# Load positive examples
positive = pd.read_csv("positive_examples.csv")

# Load hard negative examples
hard_negative = pd.read_csv("hard_negatives.csv")

# Keep the same feature columns
columns = [
    "name_similarity",
    "address_similarity",
    "country_match",
    "label"
]

positive = positive[columns]
hard_negative = hard_negative[columns]

# Combine them
training_data = pd.concat(
    [positive, hard_negative],
    ignore_index=True
)

# Shuffle the data
training_data = training_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
training_data.to_csv(
    "training_data.csv",
    index=False
)

print("\nTRAINING DATA")
print("=" * 60)

print(training_data.head(20).to_string(index=False))

print("\nTotal examples:")
print(len(training_data))

print("\nLabel distribution:")
print(training_data["label"].value_counts())

print("\nSaved as:")
print("training_data.csv")