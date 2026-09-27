import pandas as pd
import numpy as np
import unicodedata
from rapidfuzz import fuzz

print("=" * 70)
print("BUILDING MATCHING FEATURES")
print("=" * 70)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = unicodedata.normalize(
        "NFC",
        text
    )

    cleaned = []

    for char in text:

        category = unicodedata.category(char)

        if category[0] in ("L", "N", "M") or char.isspace():
            cleaned.append(char)

        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    return " ".join(text.split())


# ============================================================
# SIMILARITY
# ============================================================

def calculate_similarity(text1, text2):

    if not text1 or not text2:
        return 0.0

    return fuzz.ratio(
        text1,
        text2
    ) / 100.0


def calculate_token_similarity(text1, text2):

    if not text1 or not text2:
        return 0.0

    return fuzz.token_set_ratio(
        text1,
        text2
    ) / 100.0


# ============================================================
# LOAD SOURCE 1
# ============================================================

print()
print("Loading Source 1...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

source1 = source1.head(1000).copy()


# ============================================================
# LOAD SOURCE 2
# ============================================================

print("Loading Source 2...")

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

print("Source 2 loaded:", len(source2))


# ============================================================
# LOAD SOURCE 3
# ============================================================

print("Loading Source 3...")

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

print("Source 3 loaded:", len(source3))


# ============================================================
# LOAD CANDIDATES
# ============================================================

print()
print("Loading candidate pairs...")

candidates = pd.read_csv(
    "blocked_validation_candidates_v2.csv"
)

print(
    "Candidate pairs:",
    len(candidates)
)


# ============================================================
# CREATE LOOKUPS
# ============================================================

print()
print("Creating lookups...")

s1_lookup = source1.set_index(
    "entity_id"
)[
    ["business_name", "business_address", "country"]
].to_dict("index")


s2_lookup = source2.set_index(
    "entity_id"
)[
    ["business_name", "business_address", "country"]
].to_dict("index")


s3_lookup = source3.set_index(
    "entity_id"
)[
    ["business_name", "business_address", "country"]
].to_dict("index")


# ============================================================
# CACHE NORMALIZED TEXT
# ============================================================

print("Normalizing Source 1...")

s1_cache = {}

for entity_id, row in s1_lookup.items():

    s1_cache[entity_id] = {
        "name": normalize_text(row["business_name"]),
        "address": normalize_text(row["business_address"]),
        "country": normalize_text(row["country"])
    }


print("Normalizing Source 2...")

s2_cache = {}

for entity_id, row in s2_lookup.items():

    s2_cache[entity_id] = {
        "name": normalize_text(row["business_name"]),
        "address": normalize_text(row["business_address"]),
        "country": normalize_text(row["country"])
    }


print("Normalizing Source 3...")

s3_cache = {}

for entity_id, row in s3_lookup.items():

    s3_cache[entity_id] = {
        "name": normalize_text(row["business_name"]),
        "address": normalize_text(row["business_address"]),
        "country": normalize_text(row["country"])
    }


# ============================================================
# PROCESS IN CHUNKS
# ============================================================

CHUNK_SIZE = 100000

output_files = []

total_processed = 0


for chunk_number, start in enumerate(
    range(0, len(candidates), CHUNK_SIZE),
    start=1
):

    print()
    print(
        f"Processing chunk {chunk_number}..."
    )

    chunk = candidates.iloc[
        start:start + CHUNK_SIZE
    ].copy()

    feature_rows = []


    for _, candidate in chunk.iterrows():

        s1_id = candidate["source1_id"]
        matched_id = candidate["matched_id"]

        row1 = s1_cache.get(s1_id)

        if candidate["source"] == "S2":
            row2 = s2_cache.get(matched_id)
        else:
            row2 = s3_cache.get(matched_id)


        if row1 is None or row2 is None:
            continue


        name1 = row1["name"]
        name2 = row2["name"]

        address1 = row1["address"]
        address2 = row2["address"]

        name_similarity = calculate_similarity(
            name1,
            name2
        )

        name_token_similarity = calculate_token_similarity(
            name1,
            name2
        )

        address_similarity = calculate_similarity(
            address1,
            address2
        )

        address_token_similarity = calculate_token_similarity(
            address1,
            address2
        )

        address1_present = int(
            bool(address1)
        )

        address2_present = int(
            bool(address2)
        )

        country_match = int(
            row1["country"] != ""
            and row1["country"] == row2["country"]
        )


        feature_rows.append({

            "source1_id": s1_id,

            "matched_id": matched_id,

            "source": candidate["source"],

            "name_similarity": name_similarity,

            "name_token_similarity": name_token_similarity,

            "address_similarity": address_similarity,

            "address_token_similarity": address_token_similarity,

            "address1_present": address1_present,

            "address2_present": address2_present,

            "country_match": country_match

        })


    feature_df = pd.DataFrame(
        feature_rows
    )


    output_file = (
        f"matching_features_part_{chunk_number}.csv"
    )

    feature_df.to_csv(
        output_file,
        index=False
    )

    output_files.append(
        output_file
    )

    total_processed += len(feature_df)

    print(
        "Saved:",
        output_file
    )

    print(
        "Rows:",
        len(feature_df)
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

print()
print("=" * 70)
print("COMBINING FEATURE FILES")
print("=" * 70)

parts = []

for file in output_files:

    parts.append(
        pd.read_csv(file)
    )


features = pd.concat(
    parts,
    ignore_index=True
)


features.to_csv(
    "matching_features_v2.csv",
    index=False
)


print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()
print(
    "Total feature rows:",
    len(features)
)

print()
print(
    "Saved:",
    "matching_features_v2.csv"
)

print()
print(
    "Columns:"
)

print(
    list(features.columns)
)