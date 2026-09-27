import pandas as pd
import unicodedata
from collections import defaultdict


# =========================================================
# 1. TEXT NORMALIZATION
# =========================================================

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


def get_tokens(text):

    if not text:
        return set()

    return set(text.split())


# =========================================================
# 2. COMMON BUSINESS WORDS
# =========================================================
#
# We don't want words such as "inc" or "limited"
# to generate thousands of candidates.
#
# IMPORTANT:
# We are NOT removing these from similarity scoring.
# We are only ignoring them during blocking.
# =========================================================

COMMON_BUSINESS_WORDS = {
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "private",
    "pvt",
    "plc",
    "corp",
    "corporation",
    "company",
    "co",
    "llp",
    "lp",
    "group",
    "services",
    "service",
    "business",
    "center",
    "centre",
    "enterprises",
    "enterprise"
}


# =========================================================
# 3. LOAD SOURCE 1
# =========================================================

print("\nLoading Source 1...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=100
)

print(f"Source 1 records: {len(source1)}")


# =========================================================
# 4. LOAD SOURCE 2 SAMPLE
# =========================================================

print("\nLoading Source 2 sample...")

source2_sample = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=5000
)

print(f"Source 2 sample: {len(source2_sample)}")


# =========================================================
# 5. LOAD SOURCE 3 SAMPLE
# =========================================================

print("\nLoading Source 3 sample...")

source3_sample = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    nrows=5000
)

print(f"Source 3 sample: {len(source3_sample)}")


# =========================================================
# 6. LOAD GROUND TRUTH
# =========================================================

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


# =========================================================
# 7. FIND ALL GENUINE MATCH IDS
# =========================================================

print("\nFinding genuine match IDs...")

needed_s2_ids = set()
needed_s3_ids = set()

for source1_id in source1["entity_id"]:

    matches = ground_truth_map.get(
        source1_id,
        ""
    )

    if pd.isna(matches):
        continue

    for match_id in str(matches).split(","):

        match_id = match_id.strip()

        if match_id.startswith("S2-"):
            needed_s2_ids.add(match_id)

        elif match_id.startswith("S3-"):
            needed_s3_ids.add(match_id)


print(
    f"Genuine S2 IDs needed: {len(needed_s2_ids)}"
)

print(
    f"Genuine S3 IDs needed: {len(needed_s3_ids)}"
)


# =========================================================
# 8. SEARCH FULL SOURCE 2 FOR GENUINE MATCHES
# =========================================================

print("\nSearching Source 2 for genuine records...")

source2_true_parts = []

for chunk in pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    chunksize=100000
):

    found = chunk[
        chunk["entity_id"].isin(needed_s2_ids)
    ]

    if len(found) > 0:
        source2_true_parts.append(found)

source2_true = pd.concat(
    source2_true_parts,
    ignore_index=True
)

print(
    f"Genuine Source 2 records found: "
    f"{len(source2_true)}"
)


# =========================================================
# 9. SEARCH FULL SOURCE 3 FOR GENUINE MATCHES
# =========================================================

print("\nSearching Source 3 for genuine records...")

source3_true_parts = []

for chunk in pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    chunksize=100000
):

    found = chunk[
        chunk["entity_id"].isin(needed_s3_ids)
    ]

    if len(found) > 0:
        source3_true_parts.append(found)

source3_true = pd.concat(
    source3_true_parts,
    ignore_index=True
)

print(
    f"Genuine Source 3 records found: "
    f"{len(source3_true)}"
)


# =========================================================
# 10. BUILD TEST DATA
# =========================================================
#
# We use:
#
#   first 5000 records
#       +
#   all genuine records for our 100 Source-1 entities
#
# This lets us test whether our blocking strategies
# retain genuine matches even when those matches occur
# outside the first 5000 rows.
# =========================================================

source2 = pd.concat(
    [
        source2_sample,
        source2_true
    ],
    ignore_index=True
).drop_duplicates(
    subset=["entity_id"]
)

source3 = pd.concat(
    [
        source3_sample,
        source3_true
    ],
    ignore_index=True
).drop_duplicates(
    subset=["entity_id"]
)


print("\nFinal blocking test pools:")

print(
    f"Source 2 pool: {len(source2)}"
)

print(
    f"Source 3 pool: {len(source3)}"
)


# =========================================================
# 11. NORMALIZE DATA
# =========================================================

print("\nNormalizing data...")

for df in [source1, source2, source3]:

    df["name_normalized"] = df["business_name"].apply(
        normalize_text
    )

    df["address_normalized"] = df["business_address"].apply(
        normalize_text
    )

    df["country_normalized"] = df["country"].apply(
        normalize_text
    )

    df["name_tokens"] = df["name_normalized"].apply(
        get_tokens
    )

    df["address_tokens"] = df["address_normalized"].apply(
        get_tokens
    )


# =========================================================
# 12. CREATE BLOCKING INDEXES
# =========================================================

print("\nCreating blocking indexes...")


def build_name_index(df):

    index = defaultdict(set)

    for _, row in df.iterrows():

        country = row["country_normalized"]

        tokens = row["name_tokens"]

        for token in tokens:

            if token in COMMON_BUSINESS_WORDS:
                continue

            if len(token) < 2:
                continue

            key = (
                country,
                token
            )

            index[key].add(
                row["entity_id"]
            )

    return index


def build_address_index(df):

    index = defaultdict(set)

    for _, row in df.iterrows():

        country = row["country_normalized"]

        tokens = row["address_tokens"]

        for token in tokens:

            # Ignore extremely short address tokens
            if len(token) < 3:
                continue

            key = (
                country,
                token
            )

            index[key].add(
                row["entity_id"]
            )

    return index


def build_prefix_index(df):

    index = defaultdict(set)

    for _, row in df.iterrows():

        country = row["country_normalized"]

        name = row["name_normalized"]

        if not name:
            continue

        # First 3 characters of normalized name
        prefix = name[:3]

        if len(prefix) >= 2:

            key = (
                country,
                prefix
            )

            index[key].add(
                row["entity_id"]
            )

    return index


name_index_s2 = build_name_index(source2)
address_index_s2 = build_address_index(source2)
prefix_index_s2 = build_prefix_index(source2)

name_index_s3 = build_name_index(source3)
address_index_s3 = build_address_index(source3)
prefix_index_s3 = build_prefix_index(source3)


# =========================================================
# 13. MULTI-BLOCKING FUNCTION
# =========================================================

def generate_candidates(row, name_index, address_index, prefix_index):

    country = row["country_normalized"]

    name_tokens = row["name_tokens"]
    address_tokens = row["address_tokens"]

    candidates = set()

    # -----------------------------------------------------
    # BLOCK 1: Country + meaningful name token
    # -----------------------------------------------------

    for token in name_tokens:

        if token in COMMON_BUSINESS_WORDS:
            continue

        if len(token) < 2:
            continue

        key = (
            country,
            token
        )

        candidates.update(
            name_index.get(key, set())
        )


    # -----------------------------------------------------
    # BLOCK 2: Country + address token
    #
    # We require address tokens of length >= 3.
    # -----------------------------------------------------

    address_candidates = set()

    for token in address_tokens:

        if len(token) < 3:
            continue

        key = (
            country,
            token
        )

        address_candidates.update(
            address_index.get(key, set())
        )

    # Add address candidates
    candidates.update(address_candidates)


    # -----------------------------------------------------
    # BLOCK 3: Country + first 3 name characters
    # -----------------------------------------------------

    name = row["name_normalized"]

    if len(name) >= 2:

        prefix = name[:3]

        key = (
            country,
            prefix
        )

        candidates.update(
            prefix_index.get(key, set())
        )


    return candidates


# =========================================================
# 14. TEST ALL 100 SOURCE-1 RECORDS
# =========================================================

print("\nTesting multi-blocking...")
print("=" * 90)

total_s2_candidates = 0
total_s3_candidates = 0

total_true_s2 = 0
total_true_s3 = 0

missed_s2 = 0
missed_s3 = 0


for _, row in source1.iterrows():

    candidates_s2 = generate_candidates(
        row,
        name_index_s2,
        address_index_s2,
        prefix_index_s2
    )

    candidates_s3 = generate_candidates(
        row,
        name_index_s3,
        address_index_s3,
        prefix_index_s3
    )

    total_s2_candidates += len(candidates_s2)
    total_s3_candidates += len(candidates_s3)


    # -----------------------------------------------------
    # Get genuine matches
    # -----------------------------------------------------

    matches = ground_truth_map.get(
        row["entity_id"],
        ""
    )

    if pd.isna(matches):
        matches = ""

    true_ids = set(
        x.strip()
        for x in str(matches).split(",")
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


    # -----------------------------------------------------
    # Check recall
    # -----------------------------------------------------

    missing_s2 = true_s2 - candidates_s2
    missing_s3 = true_s3 - candidates_s3

    missed_s2 += len(missing_s2)
    missed_s3 += len(missing_s3)


    print(
        f"{row['entity_id']} | "
        f"S2 candidates: {len(candidates_s2):4d} | "
        f"S3 candidates: {len(candidates_s3):4d} | "
        f"missed S2: {len(missing_s2):2d} | "
        f"missed S3: {len(missing_s3):2d}"
    )


# =========================================================
# 15. FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 90)
print("MULTI-BLOCKING SUMMARY")
print("=" * 90)


avg_s2 = (
    total_s2_candidates
    / len(source1)
)

avg_s3 = (
    total_s3_candidates
    / len(source1)
)


print(
    f"Average S2 candidates: {avg_s2:.2f}"
)

print(
    f"Average S3 candidates: {avg_s3:.2f}"
)


print(
    f"\nTotal genuine S2 matches: {total_true_s2}"
)

print(
    f"Total genuine S3 matches: {total_true_s3}"
)


print(
    f"\nGenuine S2 matches missed: {missed_s2}"
)

print(
    f"Genuine S3 matches missed: {missed_s3}"
)


# =========================================================
# 16. CANDIDATE RECALL
# =========================================================

if total_true_s2 > 0:

    s2_recall = (
        total_true_s2 - missed_s2
    ) / total_true_s2

else:

    s2_recall = 0


if total_true_s3 > 0:

    s3_recall = (
        total_true_s3 - missed_s3
    ) / total_true_s3

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


print("\nMulti-blocking test completed.")