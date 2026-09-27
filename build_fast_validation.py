import pandas as pd
import unicodedata
from rapidfuzz import fuzz


# =========================================================
# SETTINGS
# =========================================================

NUM_SOURCE1 = 1000

# We only use a manageable sample of S2/S3 for negative
# candidate generation.
SAMPLE_SIZE = 50000

print("=" * 70)
print("BUILDING FAST BLOCKED VALIDATION DATASET")
print("=" * 70)


# =========================================================
# NORMALIZATION
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


# =========================================================
# TOKENS
# =========================================================

GENERIC_WORDS = {
    "inc", "incorporated", "llc", "ltd", "limited",
    "corp", "corporation", "company", "co",
    "pvt", "private", "plc", "lp", "llp",
    "group", "services", "service",
    "business", "enterprises", "enterprise"
}


def get_tokens(text):

    text = normalize_text(text)

    return {
        token
        for token in text.split()
        if len(token) >= 3
        and token not in GENERIC_WORDS
    }


# =========================================================
# LOAD SOURCE 1
# =========================================================

print()
print("Loading Source 1...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
).head(NUM_SOURCE1).copy()

print("Source 1:", len(source1))


# =========================================================
# LOAD GROUND TRUTH
# =========================================================

print()
print("Loading ground truth...")

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

gt = gt[
    gt["source1_entity_id"].isin(
        source1["entity_id"]
    )
].copy()

print("Ground truth:", len(gt))


# =========================================================
# LOAD S2 / S3 SAMPLE
# =========================================================

print()
print("Loading Source 2 sample...")

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=SAMPLE_SIZE
)

print("Source 2 sample:", len(source2))


print()
print("Loading Source 3 sample...")

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    nrows=SAMPLE_SIZE
)

print("Source 3 sample:", len(source3))


# =========================================================
# CREATE POSITIVE MATCH LOOKUP
# =========================================================

print()
print("Finding genuine matches...")


true_matches = {}

for _, row in gt.iterrows():

    source1_id = row["source1_entity_id"]

    value = row["matched_entity_ids"]

    if pd.isna(value):
        true_matches[source1_id] = set()
        continue

    true_matches[source1_id] = {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


# =========================================================
# FIND GENUINE MATCH ROWS
# =========================================================

print()
print("Loading exact genuine matches...")

all_ids = set()

for matches in true_matches.values():
    all_ids.update(matches)

s2_ids = {
    x for x in all_ids
    if x.startswith("S2-")
}

s3_ids = {
    x for x in all_ids
    if x.startswith("S3-")
}

# Read full S2 only when necessary, in chunks.
positive_s2 = []

if s2_ids:

    print(
        "Searching for",
        len(s2_ids),
        "genuine S2 records..."
    )

    for chunk in pd.read_csv(
        "dataset/train/train_source2.tsv",
        sep="\t",
        chunksize=100000
    ):

        found = chunk[
            chunk["entity_id"].isin(s2_ids)
        ]

        if not found.empty:
            positive_s2.append(found)

        if sum(len(x) for x in positive_s2) >= len(s2_ids):
            break


positive_s3 = []

if s3_ids:

    print(
        "Searching for",
        len(s3_ids),
        "genuine S3 records..."
    )

    for chunk in pd.read_csv(
        "dataset/train/train_source3.tsv",
        sep="\t",
        chunksize=100000
    ):

        found = chunk[
            chunk["entity_id"].isin(s3_ids)
        ]

        if not found.empty:
            positive_s3.append(found)

        if sum(len(x) for x in positive_s3) >= len(s3_ids):
            break


if positive_s2:
    positive_s2 = pd.concat(
        positive_s2,
        ignore_index=True
    )
else:
    positive_s2 = pd.DataFrame()


if positive_s3:
    positive_s3 = pd.concat(
        positive_s3,
        ignore_index=True
    )
else:
    positive_s3 = pd.DataFrame()


print(
    "Genuine S2 records found:",
    len(positive_s2)
)

print(
    "Genuine S3 records found:",
    len(positive_s3)
)


# =========================================================
# COMBINE SAMPLE + GENUINE MATCHES
# =========================================================

candidates = pd.concat(
    [
        source2,
        source3,
        positive_s2,
        positive_s3
    ],
    ignore_index=True
)

candidates = candidates.drop_duplicates(
    subset=["entity_id"]
).reset_index(drop=True)

print()
print(
    "Candidate pool:",
    len(candidates)
)


# =========================================================
# COUNTRY BLOCK
# =========================================================

source1["country_norm"] = (
    source1["country"]
    .fillna("")
    .map(normalize_text)
)

candidates["country_norm"] = (
    candidates["country"]
    .fillna("")
    .map(normalize_text)
)


# =========================================================
# PREPARE CANDIDATE TOKENS
# =========================================================

print()
print("Preparing candidate tokens...")


def first_meaningful_token(text):

    tokens = get_tokens(text)

    if not tokens:
        return ""

    return sorted(tokens)[0]


candidates["name_key"] = (
    candidates["business_name"]
    .map(first_meaningful_token)
)

candidates["address_key"] = (
    candidates["business_address"]
    .map(first_meaningful_token)
)

candidates["name_prefix"] = (
    candidates["business_name"]
    .map(normalize_text)
    .str[:3]
)


# =========================================================
# CREATE FAST LOOKUPS
# =========================================================

print("Creating lookup tables...")

name_lookup = (
    candidates[
        candidates["name_key"] != ""
    ]
    .groupby(
        ["country_norm", "name_key"]
    )["entity_id"]
    .apply(set)
    .to_dict()
)

address_lookup = (
    candidates[
        candidates["address_key"] != ""
    ]
    .groupby(
        ["country_norm", "address_key"]
    )["entity_id"]
    .apply(set)
    .to_dict()
)

prefix_lookup = (
    candidates[
        candidates["name_prefix"] != ""
    ]
    .groupby(
        ["country_norm", "name_prefix"]
    )["entity_id"]
    .apply(set)
    .to_dict()
)


candidate_lookup = candidates.set_index(
    "entity_id"
).to_dict("index")


# =========================================================
# GENERATE BLOCKED PAIRS
# =========================================================

print()
print("Generating candidates...")


records = []

total_true = 0
found_true = 0


for count, (_, row1) in enumerate(
    source1.iterrows(),
    start=1
):

    source1_id = row1["entity_id"]

    country = row1["country_norm"]

    # All true matches
    known = true_matches.get(
        source1_id,
        set()
    )

    total_true += len(known)

    candidate_ids = set()

    # -------------------------------
    # NAME TOKEN BLOCK
    # -------------------------------

    for token in get_tokens(
        row1["business_name"]
    ):

        candidate_ids.update(
            name_lookup.get(
                (country, token),
                set()
            )
        )

    # -------------------------------
    # ADDRESS TOKEN BLOCK
    # -------------------------------

    for token in get_tokens(
        row1["business_address"]
    ):

        candidate_ids.update(
            address_lookup.get(
                (country, token),
                set()
            )
        )

    # -------------------------------
    # PREFIX BLOCK
    # -------------------------------

    name = normalize_text(
        row1["business_name"]
    )

    if name:

        candidate_ids.update(
            prefix_lookup.get(
                (country, name[:3]),
                set()
            )
        )

    found_true += len(
        known.intersection(candidate_ids)
    )

    # -------------------------------
    # SAVE
    # -------------------------------

    for candidate_id in candidate_ids:

        row2 = candidate_lookup[candidate_id]

        records.append({
            "source1_id": source1_id,
            "matched_id": candidate_id,
            "source": (
                "S2"
                if candidate_id.startswith("S2-")
                else "S3"
            ),
            "business_name_1": row1["business_name"],
            "business_address_1": row1["business_address"],
            "country_1": row1["country"],
            "business_name_2": row2["business_name"],
            "business_address_2": row2["business_address"],
            "country_2": row2["country"],
            "is_true_match": int(
                candidate_id in known
            )
        })

    if count % 100 == 0:

        print(
            f"Processed {count}/{NUM_SOURCE1}..."
        )


# =========================================================
# SAVE
# =========================================================

result = pd.DataFrame(records)

result.to_csv(
    "blocked_validation_candidates.csv",
    index=False
)


# =========================================================
# RESULTS
# =========================================================

candidate_count = len(result)

recall = (
    found_true / total_true
    if total_true
    else 0
)

print()
print("=" * 70)
print("BLOCKING RESULTS")
print("=" * 70)

print("Total genuine matches:", total_true)

print("Genuine matches found:", found_true)

print(
    "Genuine matches missed:",
    total_true - found_true
)

print(
    "Candidate pairs:",
    candidate_count
)

print(
    "Candidate recall:",
    round(recall, 4)
)

print(
    "Average candidates per Source 1:",
    round(
        candidate_count / NUM_SOURCE1,
        2
    )
)

print()
print("Saved:")
print("blocked_validation_candidates.csv")

print()
print("=" * 70)
print("DONE")
print("=" * 70)