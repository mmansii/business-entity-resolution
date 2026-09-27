import pandas as pd
import unicodedata

# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = unicodedata.normalize("NFC", text)

    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if category[0] in ("L", "N", "M") or char.isspace():
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    return " ".join(text.split())


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("\nLoading Source 1...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=100
)

print(f"Source 1 records: {len(source1)}")


print("\nLoading Source 2...")

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=5000
)

print(f"Source 2 records: {len(source2)}")


print("\nLoading Source 3...")

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    nrows=5000
)

print(f"Source 3 records: {len(source3)}")


# ---------------------------------------------------------
# Normalize country and names
# ---------------------------------------------------------

source1["country_normalized"] = source1["country"].apply(
    normalize_text
)

source2["country_normalized"] = source2["country"].apply(
    normalize_text
)

source3["country_normalized"] = source3["country"].apply(
    normalize_text
)


source1["name_normalized"] = source1["business_name"].apply(
    normalize_text
)

source2["name_normalized"] = source2["business_name"].apply(
    normalize_text
)

source3["name_normalized"] = source3["business_name"].apply(
    normalize_text
)


# ---------------------------------------------------------
# Create name token sets
# ---------------------------------------------------------

def get_tokens(text):

    if not text:
        return set()

    return set(text.split())


source1["name_tokens"] = source1["name_normalized"].apply(
    get_tokens
)

source2["name_tokens"] = source2["name_normalized"].apply(
    get_tokens
)

source3["name_tokens"] = source3["name_normalized"].apply(
    get_tokens
)


# ---------------------------------------------------------
# Country + name-token candidate generation
# ---------------------------------------------------------

def generate_candidates(source1_row, source_df):

    country = source1_row["country_normalized"]
    source1_tokens = source1_row["name_tokens"]

    candidates = []

    for _, row in source_df.iterrows():

        # STEP 1:
        # Country must match
        if row["country_normalized"] != country:
            continue

        # STEP 2:
        # Ignore empty names
        candidate_tokens = row["name_tokens"]

        if not source1_tokens or not candidate_tokens:
            continue

        # STEP 3:
        # At least one name token must overlap
        common_tokens = source1_tokens.intersection(
            candidate_tokens
        )

        if len(common_tokens) >= 1:

            candidates.append(row["entity_id"])

    return candidates


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

print("\nTesting country + name blocking...")
print("=" * 80)

total_candidates_s2 = 0
total_candidates_s3 = 0

for _, row in source1.head(10).iterrows():

    candidates_s2 = generate_candidates(
        row,
        source2
    )

    candidates_s3 = generate_candidates(
        row,
        source3
    )

    total_candidates_s2 += len(candidates_s2)
    total_candidates_s3 += len(candidates_s3)

    print("\nSource 1:")
    print(row["entity_id"])
    print("Business:", row["business_name"])
    print("Country:", row["country"])

    print(
        f"Source 2 candidates: {len(candidates_s2)}"
    )

    print(
        f"Source 3 candidates: {len(candidates_s3)}"
    )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("COUNTRY + NAME BLOCKING SUMMARY")
print("=" * 80)

print(
    f"Average Source 2 candidates: "
    f"{total_candidates_s2 / 10:.2f}"
)

print(
    f"Average Source 3 candidates: "
    f"{total_candidates_s3 / 10:.2f}"
)

print("\nCountry + name blocking test completed.")