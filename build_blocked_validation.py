import pandas as pd
import unicodedata
from collections import defaultdict
from rapidfuzz import fuzz


# =========================================================
# SETTINGS
# =========================================================

NUM_SOURCE1 = 1000

print("=" * 70)
print("BUILDING BLOCKED VALIDATION DATASET")
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
# TOKENIZATION
# =========================================================

GENERIC_WORDS = {
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "company",
    "co",
    "pvt",
    "private",
    "plc",
    "lp",
    "llp",
    "group",
    "services",
    "service",
    "business",
    "enterprises",
    "enterprise",
}


def get_tokens(text):

    normalized = normalize_text(text)

    return set(
        token
        for token in normalized.split()
        if token not in GENERIC_WORDS
        and len(token) >= 3
    )


# =========================================================
# LOAD SOURCE 1
# =========================================================

print()
print("Loading Source 1...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

source1 = source1.head(NUM_SOURCE1).copy()

print("Source 1 entities:", len(source1))


# =========================================================
# LOAD GROUND TRUTH
# =========================================================

print()
print("Loading ground truth...")

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

ground_truth = ground_truth[
    ground_truth["source1_entity_id"].isin(
        source1["entity_id"]
    )
].copy()

print("Ground truth rows:", len(ground_truth))


# =========================================================
# LOAD SOURCE 2 AND SOURCE 3
# =========================================================

print()
print("Loading Source 2...")

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

print("Source 2 rows:", len(source2))


print()
print("Loading Source 3...")

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

print("Source 3 rows:", len(source3))


# =========================================================
# PRECOMPUTE NORMALIZED FIELDS
# =========================================================

print()
print("Normalizing data...")

for df in [source1, source2, source3]:

    df["norm_name"] = df["business_name"].apply(
        normalize_text
    )

    df["norm_address"] = df["business_address"].apply(
        normalize_text
    )

    df["norm_country"] = df["country"].fillna("").apply(
        normalize_text
    )


# =========================================================
# BUILD BLOCK INDEXES
# =========================================================

print()
print("Building blocking indexes...")


# ---------------------------------------------------------
# Name token index
# ---------------------------------------------------------

name_index = defaultdict(set)

for idx, row in pd.concat(
    [source2, source3],
    ignore_index=True
).iterrows():

    tokens = get_tokens(row["business_name"])

    for token in tokens:

        name_index[
            (row["norm_country"], token)
        ].add(idx)


# ---------------------------------------------------------
# Address token index
# ---------------------------------------------------------

address_index = defaultdict(set)

for idx, row in pd.concat(
    [source2, source3],
    ignore_index=True
).iterrows():

    address_tokens = get_tokens(
        row["business_address"]
    )

    for token in address_tokens:

        address_index[
            (row["norm_country"], token)
        ].add(idx)


# ---------------------------------------------------------
# Name prefix index
# ---------------------------------------------------------

prefix_index = defaultdict(set)

for idx, row in pd.concat(
    [source2, source3],
    ignore_index=True
).iterrows():

    name = row["norm_name"]

    if name:

        prefix = name[:3]

        prefix_index[
            (row["norm_country"], prefix)
        ].add(idx)


print("Blocking indexes ready.")


# =========================================================
# COMBINE SOURCE 2 + SOURCE 3
# =========================================================

all_candidates = pd.concat(
    [source2, source3],
    ignore_index=True
)

print()
print("Combined candidate records:", len(all_candidates))


# =========================================================
# GROUND TRUTH LOOKUP
# =========================================================

print()
print("Building ground-truth lookup...")

true_matches = {}

for _, row in ground_truth.iterrows():

    source1_id = row["source1_entity_id"]

    matched_ids = row["matched_entity_ids"]

    if pd.isna(matched_ids):

        true_matches[source1_id] = set()

        continue

    matched_ids = str(matched_ids).strip()

    if not matched_ids:

        true_matches[source1_id] = set()

        continue

    true_matches[source1_id] = set(
        x.strip()
        for x in matched_ids.split(",")
        if x.strip()
    )


# =========================================================
# CREATE CANDIDATES
# =========================================================

print()
print("Generating blocked candidate pairs...")

candidate_pairs = []

total_genuine = 0
missed_genuine = 0


for counter, (_, row1) in enumerate(
    source1.iterrows(),
    start=1
):

    source1_id = row1["entity_id"]

    country = row1["norm_country"]

    # -----------------------------------------------------
    # Candidate set for this Source 1 record
    # -----------------------------------------------------

    candidate_indices = set()

    # Name-token blocks
    name_tokens = get_tokens(
        row1["business_name"]
    )

    for token in name_tokens:

        candidate_indices.update(
            name_index.get(
                (country, token),
                set()
            )
        )

    # Address-token blocks
    address_tokens = get_tokens(
        row1["business_address"]
    )

    for token in address_tokens:

        candidate_indices.update(
            address_index.get(
                (country, token),
                set()
            )
        )

    # Name-prefix block
    name = row1["norm_name"]

    if name:

        prefix = name[:3]

        candidate_indices.update(
            prefix_index.get(
                (country, prefix),
                set()
            )
        )

    # -----------------------------------------------------
    # Genuine matches
    # -----------------------------------------------------

    known_matches = true_matches.get(
        source1_id,
        set()
    )

    total_genuine += len(known_matches)

    found_ids = set()

    # -----------------------------------------------------
    # Store candidates
    # -----------------------------------------------------

    for idx in candidate_indices:

        row2 = all_candidates.iloc[idx]

        matched_id = row2["entity_id"]

        found_ids.add(matched_id)

        candidate_pairs.append({
            "source1_id": source1_id,
            "matched_id": matched_id,
            "source": (
                "S2"
                if matched_id.startswith("S2-")
                else "S3"
            ),
            "business_name_1": row1["business_name"],
            "business_address_1": row1["business_address"],
            "country_1": row1["country"],
            "business_name_2": row2["business_name"],
            "business_address_2": row2["business_address"],
            "country_2": row2["country"],
            "is_true_match": int(
                matched_id in known_matches
            )
        })

    # -----------------------------------------------------
    # Check missed genuine matches
    # -----------------------------------------------------

    missed = known_matches - found_ids

    missed_genuine += len(missed)

    if missed:

        print()
        print(
            "WARNING - missed genuine match:"
        )

        print(
            source1_id,
            "->",
            missed
        )

    # Progress
    if counter % 100 == 0:

        print(
            f"Processed {counter}/{NUM_SOURCE1} "
            f"Source 1 records..."
        )


# =========================================================
# SAVE CANDIDATES
# =========================================================

print()
print("Creating candidate DataFrame...")

candidate_df = pd.DataFrame(
    candidate_pairs
)

candidate_df.to_csv(
    "blocked_validation_candidates.csv",
    index=False
)


# =========================================================
# STATISTICS
# =========================================================

total_candidates = len(candidate_df)

true_candidates = candidate_df[
    candidate_df["is_true_match"] == 1
]

candidate_recall = (
    len(true_candidates) / total_genuine
    if total_genuine > 0
    else 0
)


print()
print("=" * 70)
print("BLOCKING RESULTS")
print("=" * 70)

print(
    "Total genuine matches:",
    total_genuine
)

print(
    "Genuine matches found:",
    len(true_candidates)
)

print(
    "Genuine matches missed:",
    missed_genuine
)

print(
    "Candidate pairs:",
    total_candidates
)

print(
    "Candidate recall:",
    round(candidate_recall, 4)
)

print(
    "Average candidates per Source 1:",
    round(
        total_candidates / NUM_SOURCE1,
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