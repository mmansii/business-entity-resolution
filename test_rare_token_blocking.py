import pandas as pd
import unicodedata
from collections import Counter

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
# Token helper
# ---------------------------------------------------------

def get_tokens(text):

    if not text:
        return set()

    return set(text.split())


# ---------------------------------------------------------
# Load Source 1
# ---------------------------------------------------------

print("\nLoading Source 1...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=100
)

print(f"Source 1 records: {len(source1)}")


# ---------------------------------------------------------
# Load Source 2
# ---------------------------------------------------------

print("\nLoading Source 2...")

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=5000
)

print(f"Source 2 records: {len(source2)}")


# ---------------------------------------------------------
# Load Source 3
# ---------------------------------------------------------

print("\nLoading Source 3...")

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    nrows=5000
)

print(f"Source 3 records: {len(source3)}")


# ---------------------------------------------------------
# Normalize names and countries
# ---------------------------------------------------------

print("\nNormalizing data...")

for df in [source1, source2, source3]:

    df["name_normalized"] = df["business_name"].apply(
        normalize_text
    )

    df["country_normalized"] = df["country"].apply(
        normalize_text
    )

    df["name_tokens"] = df["name_normalized"].apply(
        get_tokens
    )


# ---------------------------------------------------------
# Build token frequency from Source 2 + Source 3
# ---------------------------------------------------------

print("\nBuilding token frequency index...")

token_frequency = Counter()

for df in [source2, source3]:

    for tokens in df["name_tokens"]:

        for token in tokens:

            token_frequency[token] += 1


print(
    f"Unique name tokens found: {len(token_frequency)}"
)


# ---------------------------------------------------------
# Display common and rare tokens
# ---------------------------------------------------------

print("\nMost common tokens:")
print("-" * 60)

for token, count in token_frequency.most_common(20):

    print(
        f"{token:<20} {count}"
    )


# ---------------------------------------------------------
# Rare token definition
# ---------------------------------------------------------
#
# A token appearing at most 20 times is considered rare.
#
# We will test this threshold first.
# ---------------------------------------------------------

RARE_TOKEN_MAX_FREQUENCY = 20


# ---------------------------------------------------------
# Create country + rare-token index
# ---------------------------------------------------------

print("\nBuilding candidate index...")

# Structure:
#
# country -> token -> set(entity IDs)

candidate_index_s2 = {}
candidate_index_s3 = {}


def build_index(df):

    index = {}

    for _, row in df.iterrows():

        country = row["country_normalized"]

        if country not in index:
            index[country] = {}

        for token in row["name_tokens"]:

            # Only index rare tokens
            if token_frequency[token] <= RARE_TOKEN_MAX_FREQUENCY:

                if token not in index[country]:
                    index[country][token] = set()

                index[country][token].add(
                    row["entity_id"]
                )

    return index


candidate_index_s2 = build_index(source2)
candidate_index_s3 = build_index(source3)


# ---------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------

def generate_candidates(row, index):

    country = row["country_normalized"]
    tokens = row["name_tokens"]

    candidates = set()

    if country not in index:
        return candidates

    country_index = index[country]

    for token in tokens:

        # Only use rare tokens
        if token_frequency[token] <= RARE_TOKEN_MAX_FREQUENCY:

            if token in country_index:

                candidates.update(
                    country_index[token]
                )

    return candidates


# ---------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------

print("\nLoading ground truth...")

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

ground_truth_map = dict(
    zip(
        ground_truth["source1_entity_id"],
        ground_truth["matched_entity_ids"]
    )
)


# ---------------------------------------------------------
# Test candidate generation
# ---------------------------------------------------------

print("\nTesting rare-token candidate generation...")
print("=" * 90)

total_s2 = 0
total_s3 = 0

total_true_s2 = 0
total_true_s3 = 0

missed_s2 = 0
missed_s3 = 0


for _, row in source1.iterrows():

    candidates_s2 = generate_candidates(
        row,
        candidate_index_s2
    )

    candidates_s3 = generate_candidates(
        row,
        candidate_index_s3
    )

    total_s2 += len(candidates_s2)
    total_s3 += len(candidates_s3)

    # ---------------------------------------------
    # Get genuine matches for this Source 1 entity
    # ---------------------------------------------

    true_matches = ground_truth_map.get(
        row["entity_id"],
        ""
    )

    if pd.isna(true_matches):
        true_matches = ""

    true_matches = str(true_matches)

    true_ids = set(
        x.strip()
        for x in true_matches.split(",")
        if x.strip()
    )

    true_s2 = {
        x for x in true_ids
        if x.startswith("S2-")
    }

    true_s3 = {
        x for x in true_ids
        if x.startswith("S3-")
    }

    total_true_s2 += len(true_s2)
    total_true_s3 += len(true_s3)

    # Check whether every genuine match survived blocking

    missing_s2 = true_s2 - candidates_s2
    missing_s3 = true_s3 - candidates_s3

    missed_s2 += len(missing_s2)
    missed_s3 += len(missing_s3)

    print("\nSource 1:")
    print(row["entity_id"])
    print("Business:", row["business_name"])
    print("Country:", row["country"])

    print(
        f"S2 candidates: {len(candidates_s2)}"
    )

    print(
        f"S3 candidates: {len(candidates_s3)}"
    )

    print(
        f"Genuine S2 matches: {len(true_s2)}"
    )

    print(
        f"Genuine S3 matches: {len(true_s3)}"
    )

    print(
        f"Missed genuine S2 matches: {len(missing_s2)}"
    )

    print(
        f"Missed genuine S3 matches: {len(missing_s3)}"
    )


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n")
print("=" * 90)
print("RARE TOKEN BLOCKING SUMMARY")
print("=" * 90)

print(
    f"Average S2 candidates: "
    f"{total_s2 / len(source1):.2f}"
)

print(
    f"Average S3 candidates: "
    f"{total_s3 / len(source1):.2f}"
)

print(
    f"\nTotal genuine S2 matches: "
    f"{total_true_s2}"
)

print(
    f"Total genuine S3 matches: "
    f"{total_true_s3}"
)

print(
    f"\nGenuine S2 matches missed by blocking: "
    f"{missed_s2}"
)

print(
    f"Genuine S3 matches missed by blocking: "
    f"{missed_s3}"
)

# ---------------------------------------------------------
# Recall of candidate generation
# ---------------------------------------------------------

if total_true_s2 > 0:

    s2_recall = (
        (total_true_s2 - missed_s2)
        / total_true_s2
    )

else:

    s2_recall = 0


if total_true_s3 > 0:

    s3_recall = (
        (total_true_s3 - missed_s3)
        / total_true_s3
    )

else:

    s3_recall = 0


print(
    f"\nS2 candidate recall: "
    f"{s2_recall:.3f}"
)

print(
    f"S3 candidate recall: "
    f"{s3_recall:.3f}"
)

print("\nRare-token blocking test completed.")