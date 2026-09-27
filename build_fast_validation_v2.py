import pandas as pd
import re
import unicodedata
from collections import defaultdict

print("=" * 70)
print("BUILDING IMPROVED BLOCKED VALIDATION")
print("=" * 70)


# ============================================================
# SETTINGS
# ============================================================

NUM_SOURCE1 = 1000
SAMPLE_SIZE = 50000


# ============================================================
# NORMALIZATION
# ============================================================

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


# ============================================================
# GENERIC WORDS
# ============================================================

GENERIC_NAME_WORDS = {
    "private",
    "limited",
    "ltd",
    "llp",
    "llc",
    "inc",
    "incorporated",
    "company",
    "co",
    "corporation",
    "corp",
    "pvt",
    "plc",
}


GENERIC_ADDRESS_WORDS = {
    "road",
    "street",
    "st",
    "rd",
    "avenue",
    "ave",
    "lane",
    "ln",
    "drive",
    "dr",
    "floor",
    "fl",
    "building",
    "bldg",
    "city",
    "state",
    "county",
    "district",
    "area",
    "main",
    "side",
    "front",
    "india",
    "maharashtra",
    "delhi",
    "mumbai",
    "bangalore",
    "karnataka",
    "uttar",
    "pradesh",
}


# ============================================================
# TOKEN FUNCTIONS
# ============================================================

def get_name_tokens(text):

    normalized = normalize_text(text)

    tokens = normalized.split()

    return {
        token
        for token in tokens
        if len(token) >= 3
        and token not in GENERIC_NAME_WORDS
    }


def get_address_tokens(text):

    normalized = normalize_text(text)

    tokens = normalized.split()

    result = set()

    for token in tokens:

        if len(token) < 3:
            continue

        if token in GENERIC_ADDRESS_WORDS:
            continue

        result.add(token)

    return result


def get_address_numbers(text):

    normalized = normalize_text(text)

    tokens = normalized.split()

    numbers = set()

    for token in tokens:

        # Pure numbers
        if token.isdigit():
            numbers.add(token)

        # Things like 522a, d220, 26a
        elif re.search(r"\d", token):
            numbers.add(token)

    return numbers


# ============================================================
# LOAD SOURCE 1
# ============================================================

print()
print("Loading Source 1...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

source1 = source1.head(NUM_SOURCE1).copy()


# ============================================================
# LOAD SOURCE 2
# ============================================================

print("Loading Source 2 sample...")

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=SAMPLE_SIZE
)


# ============================================================
# LOAD SOURCE 3
# ============================================================

print("Loading Source 3 sample...")

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    nrows=SAMPLE_SIZE
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

ground_truth = ground_truth[
    ground_truth["source1_entity_id"].isin(source1["entity_id"])
].copy()


# ============================================================
# GET EXACT POSITIVE RECORDS
# ============================================================

print()
print("Finding exact positive records...")


positive_ids_s2 = set()
positive_ids_s3 = set()

for matches in ground_truth["matched_entity_ids"].dropna():

    for entity_id in str(matches).split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            positive_ids_s2.add(entity_id)

        elif entity_id.startswith("S3-"):
            positive_ids_s3.add(entity_id)


print("Positive S2 IDs:", len(positive_ids_s2))
print("Positive S3 IDs:", len(positive_ids_s3))


# ============================================================
# ADD POSITIVE RECORDS NOT IN SAMPLE
# ============================================================

print()
print("Scanning full S2 for missing positive records...")


extra_s2 = []

for chunk in pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    chunksize=100000
):

    found = chunk[
        chunk["entity_id"].isin(positive_ids_s2)
    ]

    if len(found) > 0:
        extra_s2.append(found)


if extra_s2:

    extra_s2 = pd.concat(extra_s2, ignore_index=True)

    source2 = pd.concat(
        [source2, extra_s2],
        ignore_index=True
    )


print("Scanning full S3 for missing positive records...")


extra_s3 = []

for chunk in pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    chunksize=100000
):

    found = chunk[
        chunk["entity_id"].isin(positive_ids_s3)
    ]

    if len(found) > 0:
        extra_s3.append(found)


if extra_s3:

    extra_s3 = pd.concat(extra_s3, ignore_index=True)

    source3 = pd.concat(
        [source3, extra_s3],
        ignore_index=True
    )


# Remove duplicates

source2 = source2.drop_duplicates(
    subset=["entity_id"]
).reset_index(drop=True)

source3 = source3.drop_duplicates(
    subset=["entity_id"]
).reset_index(drop=True)


print()
print("S2 candidate pool:", len(source2))
print("S3 candidate pool:", len(source3))


# ============================================================
# CREATE BLOCK INDEX
# ============================================================

def build_indexes(df):

    indexes = {

        "name_token": defaultdict(set),

        "address_token": defaultdict(set),

        "name_prefix": defaultdict(set),

        "address_number": defaultdict(set),

    }


    for _, row in df.iterrows():

        entity_id = row["entity_id"]

        country = normalize_text(row["country"])

        # -------------------------
        # NAME TOKENS
        # -------------------------

        name_tokens = get_name_tokens(
            row["business_name"]
        )

        for token in name_tokens:

            indexes["name_token"][
                (country, token)
            ].add(entity_id)


        # -------------------------
        # ADDRESS TOKENS
        # -------------------------

        address_tokens = get_address_tokens(
            row["business_address"]
        )

        for token in address_tokens:

            indexes["address_token"][
                (country, token)
            ].add(entity_id)


        # -------------------------
        # NAME PREFIX
        # -------------------------

        name = normalize_text(
            row["business_name"]
        )

        if name:

            prefix = name[:3]

            indexes["name_prefix"][
                (country, prefix)
            ].add(entity_id)


        # -------------------------
        # ADDRESS NUMBERS
        # -------------------------

        numbers = get_address_numbers(
            row["business_address"]
        )

        for number in numbers:

            indexes["address_number"][
                (country, number)
            ].add(entity_id)


    return indexes


print()
print("Building S2 indexes...")

s2_indexes = build_indexes(source2)


print("Building S3 indexes...")

s3_indexes = build_indexes(source3)


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(row, indexes):

    country = normalize_text(
        row["country"]
    )

    candidates = set()


    # ========================================================
    # BLOCK 1: NAME TOKEN
    # ========================================================

    name_tokens = get_name_tokens(
        row["business_name"]
    )

    for token in name_tokens:

        candidates.update(
            indexes["name_token"].get(
                (country, token),
                set()
            )
        )


    # ========================================================
    # BLOCK 2: ADDRESS TOKEN
    # ========================================================

    address_tokens = get_address_tokens(
        row["business_address"]
    )

    for token in address_tokens:

        candidates.update(
            indexes["address_token"].get(
                (country, token),
                set()
            )
        )


    # ========================================================
    # BLOCK 3: NAME PREFIX
    # ========================================================

    name = normalize_text(
        row["business_name"]
    )

    if name:

        prefix = name[:3]

        candidates.update(
            indexes["name_prefix"].get(
                (country, prefix),
                set()
            )
        )


    # ========================================================
    # BLOCK 4: ADDRESS NUMBER
    # ========================================================

    numbers = get_address_numbers(
        row["business_address"]
    )

    for number in numbers:

        candidates.update(
            indexes["address_number"].get(
                (country, number),
                set()
            )
        )


    return candidates


# ============================================================
# GENERATE ALL CANDIDATE PAIRS
# ============================================================

print()
print("Generating S2 candidates...")

candidate_rows = []


for _, row in source1.iterrows():

    candidates = generate_candidates(
        row,
        s2_indexes
    )

    for candidate_id in candidates:

        candidate_rows.append({
            "source1_id": row["entity_id"],
            "matched_id": candidate_id,
            "source": "S2"
        })


print("Generating S3 candidates...")


for _, row in source1.iterrows():

    candidates = generate_candidates(
        row,
        s3_indexes
    )

    for candidate_id in candidates:

        candidate_rows.append({
            "source1_id": row["entity_id"],
            "matched_id": candidate_id,
            "source": "S3"
        })


# ============================================================
# SAVE
# ============================================================

candidates_df = pd.DataFrame(candidate_rows)

candidates_df = candidates_df.drop_duplicates(
    subset=["source1_id", "matched_id"]
)


candidates_df.to_csv(
    "blocked_validation_candidates_v2.csv",
    index=False
)


# ============================================================
# CHECK RECALL
# ============================================================

candidate_pairs = set(
    zip(
        candidates_df["source1_id"],
        candidates_df["matched_id"]
    )
)


total_positive = 0
found_positive = 0


for _, row in ground_truth.iterrows():

    source1_id = row["source1_entity_id"]

    if pd.isna(row["matched_entity_ids"]):
        continue

    for matched_id in str(
        row["matched_entity_ids"]
    ).split(","):

        matched_id = matched_id.strip()

        if not matched_id:
            continue

        total_positive += 1

        if (
            source1_id,
            matched_id
        ) in candidate_pairs:

            found_positive += 1


print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print()
print("Total genuine matches:", total_positive)

print(
    "Found by blocking:",
    found_positive
)

print(
    "Missed:",
    total_positive - found_positive
)

print(
    "Candidate recall:",
    round(
        found_positive / total_positive,
        4
    )
)

print(
    "Total candidate pairs:",
    len(candidates_df)
)

print(
    "Average candidates per Source 1:",
    round(
        len(candidates_df) / len(source1),
        2
    )
)

print()
print("Saved:")
print("blocked_validation_candidates_v2.csv")