import pandas as pd


# Load small samples first
source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=10000
)

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=10000
)


def normalize_name(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Keep letters/numbers and turn punctuation into spaces
    cleaned = []

    for char in text:
        if char.isalnum() or char.isspace():
            cleaned.append(char)
        else:
            cleaned.append(" ")

    return " ".join("".join(cleaned).split())


# Normalize names
source1["name_normalized"] = source1["business_name"].apply(normalize_name)
source2["name_normalized"] = source2["business_name"].apply(normalize_name)

# Normalize country
source1["country"] = source1["country"].fillna("").str.lower().str.strip()
source2["country"] = source2["country"].fillna("").str.lower().str.strip()


# Pick first Source 1 business
business = source1.iloc[0]

print("SOURCE 1 BUSINESS")
print("------------------")
print("ID:", business["entity_id"])
print("Name:", business["business_name"])
print("Country:", business["country"])


# Step 1: country blocking
candidates = source2[
    source2["country"] == business["country"]
].copy()

print("\nAfter country blocking:", len(candidates))


# Step 2: extract useful words from Source 1 name
tokens = [
    word
    for word in business["name_normalized"].split()
    if len(word) >= 3
]

print("Useful name tokens:", tokens)


# Find candidates sharing at least one useful name token
if tokens:
    pattern = "|".join(
        [rf"\b{token}\b" for token in tokens]
    )

    name_candidates = candidates[
        candidates["name_normalized"].str.contains(
            pattern,
            regex=True,
            na=False
        )
    ]
else:
    name_candidates = candidates.iloc[0:0]


print("After name blocking:", len(name_candidates))

print("\nName-based candidates:")
print(
    name_candidates[
        ["entity_id", "business_name", "country"]
    ].head(20).to_string(index=False)
)