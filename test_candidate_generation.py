import pandas as pd
import unicodedata
import re

# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Keep Hindi, English, numbers and combining marks
    text = unicodedata.normalize("NFC", text)

    cleaned = []

    for char in text:

        category = unicodedata.category(char)

        if category[0] in ("L", "N", "M") or char.isspace():
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    text = " ".join(text.split())

    return text


# ---------------------------------------------------------
# Load the small validation Source 1 dataset
# ---------------------------------------------------------

print("\nLoading Source 1...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=100
)

print(f"Source 1 records: {len(source1)}")


# ---------------------------------------------------------
# Load Source 2 and Source 3 sample
# ---------------------------------------------------------

print("\nLoading Source 2 sample...")

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=5000
)

print(f"Source 2 records: {len(source2)}")


print("\nLoading Source 3 sample...")

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    nrows=5000
)

print(f"Source 3 records: {len(source3)}")


# ---------------------------------------------------------
# Normalize names
# ---------------------------------------------------------

print("\nNormalizing business names...")

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
# Create token sets
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
# Candidate generation using name tokens
# ---------------------------------------------------------

def generate_candidates(source1_row, source_df):

    source1_tokens = source1_row["name_tokens"]

    candidates = []

    for _, row in source_df.iterrows():

        candidate_tokens = row["name_tokens"]

        # Ignore empty names
        if not source1_tokens or not candidate_tokens:
            continue

        # Find common words
        common_tokens = source1_tokens.intersection(
            candidate_tokens
        )

        # At least one common token
        if len(common_tokens) >= 1:

            candidates.append(row["entity_id"])

    return candidates


# ---------------------------------------------------------
# Test candidate generation
# ---------------------------------------------------------

print("\nTesting candidate generation...")
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
    print(row["business_name"])

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
print("CANDIDATE GENERATION SUMMARY")
print("=" * 80)

print(
    f"Average Source 2 candidates: "
    f"{total_candidates_s2 / 10:.2f}"
)

print(
    f"Average Source 3 candidates: "
    f"{total_candidates_s3 / 10:.2f}"
)

print("\nCandidate generation test completed.")