import pandas as pd
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


TRAIN_DIR = Path("dataset/train")

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"


# ============================================================
# CAPS
# ============================================================

NAME_CAP = 100
ADDRESS_CAP = 250
PREFIX_CAP = 250
NUMBER_CAP = 250


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

    return " ".join("".join(cleaned).split())


def meaningful_tokens(text):

    stopwords = {
        "the", "and", "of", "for", "inc", "llc", "ltd",
        "limited", "company", "co", "corp", "corporation",
        "private", "pvt", "plc", "llp", "india", "ind",
        "services", "service", "business", "group",
        "center", "centre", "international"
    }

    return [
        token
        for token in text.split()
        if len(token) >= 3 and token not in stopwords
    ]


def extract_numbers(text):

    if not text:
        return []

    return re.findall(r"\d+[a-z]?", text)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("Loading training data...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str
).fillna("")

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str
).fillna("")

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    dtype=str
).fillna("")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
).fillna("")

print(f"S1: {len(s1):,}")
print(f"S2: {len(s2):,}")
print(f"S3: {len(s3):,}")


# ============================================================
# NORMALIZE
# ============================================================

for df in [s1, s2, s3]:

    df["name_norm"] = df["business_name"].map(normalize_text)
    df["address_norm"] = df["business_address"].map(normalize_text)
    df["country_norm"] = df["country"].map(normalize_text)


# ============================================================
# BUILD INDEX WITH CAPS
# ============================================================

def build_index(df):

    token_temp = defaultdict(list)
    address_temp = defaultdict(list)
    prefix_temp = defaultdict(list)
    number_temp = defaultdict(list)

    for idx, row in enumerate(df.itertuples(index=False)):

        country = row.country_norm

        if not country:
            continue

        # NAME TOKENS

        for token in set(meaningful_tokens(row.name_norm)):

            token_temp[(country, token)].append(idx)

        # ADDRESS TOKENS

        for token in set(meaningful_tokens(row.address_norm)):

            address_temp[(country, token)].append(idx)

        # NAME PREFIX

        compact = row.name_norm.replace(" ", "")

        if len(compact) >= 3:

            prefix_temp[
                (country, compact[:3])
            ].append(idx)

        # ADDRESS NUMBERS

        for number in set(extract_numbers(row.address_norm)):

            number_temp[
                (country, number)
            ].append(idx)

    # Keep only blocks below the cap.

    token_index = {
        key: value
        for key, value in token_temp.items()
        if len(value) <= NAME_CAP
    }

    address_index = {
        key: value
        for key, value in address_temp.items()
        if len(value) <= ADDRESS_CAP
    }

    prefix_index = {
        key: value
        for key, value in prefix_temp.items()
        if len(value) <= PREFIX_CAP
    }

    number_index = {
        key: value
        for key, value in number_temp.items()
        if len(value) <= NUMBER_CAP
    }

    return (
        token_index,
        address_index,
        prefix_index,
        number_index
    )


print("\nBuilding S2 index...")

s2_index = build_index(s2)

print("Building S3 index...")

s3_index = build_index(s3)


# ============================================================
# CANDIDATES
# ============================================================

def get_candidates(
    row,
    indexes
):

    token_index, address_index, prefix_index, number_index = indexes

    country = row.country_norm
    name = row.name_norm
    address = row.address_norm

    candidates = set()

    # NAME TOKEN

    for token in set(meaningful_tokens(name)):

        candidates.update(
            token_index.get(
                (country, token),
                []
            )
        )

    # ADDRESS TOKEN

    for token in set(meaningful_tokens(address)):

        candidates.update(
            address_index.get(
                (country, token),
                []
            )
        )

    # PREFIX

    compact = name.replace(" ", "")

    if len(compact) >= 3:

        candidates.update(
            prefix_index.get(
                (country, compact[:3]),
                []
            )
        )

    # NUMBER

    for number in set(extract_numbers(address)):

        candidates.update(
            number_index.get(
                (country, number),
                []
            )
        )

    return candidates


# ============================================================
# GROUND TRUTH
# ============================================================

gt_map = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_entity_ids"]
    )
)


# ============================================================
# TEST FIRST 1000 S1
# ============================================================

TEST_COUNT = 1000

total_true = 0
found_true_s2 = 0
found_true_s3 = 0

total_candidates = 0

print()
print("=" * 70)
print("TESTING BLOCKING")
print("=" * 70)

for i in range(min(TEST_COUNT, len(s1))):

    row = s1.iloc[i]

    source1_id = row["entity_id"]

    true_matches = gt_map.get(
        source1_id,
        ""
    )

    true_ids = set()

    if true_matches:

        true_ids = set(
            true_matches.split(",")
        )

    true_s2 = {
        x for x in true_ids
        if x.startswith("S2-")
    }

    true_s3 = {
        x for x in true_ids
        if x.startswith("S3-")
    }

    total_true += len(true_ids)

    # S2

    candidates = get_candidates(
        row,
        s2_index
    )

    candidate_ids = {
        s2.iloc[x]["entity_id"]
        for x in candidates
    }

    found_true_s2 += len(
        true_s2 & candidate_ids
    )

    total_candidates += len(candidate_ids)

    # S3

    candidates = get_candidates(
        row,
        s3_index
    )

    candidate_ids = {
        s3.iloc[x]["entity_id"]
        for x in candidates
    }

    found_true_s3 += len(
        true_s3 & candidate_ids
    )

    total_candidates += len(candidate_ids)

    if (i + 1) % 100 == 0:

        print(
            f"Checked {i + 1:,}/{TEST_COUNT:,}"
        )


# ============================================================
# RESULTS
# ============================================================

total_true_s2 = 0
total_true_s3 = 0

for i in range(min(TEST_COUNT, len(s1))):

    source1_id = s1.iloc[i]["entity_id"]

    true_matches = gt_map.get(
        source1_id,
        ""
    )

    if true_matches:

        ids = set(true_matches.split(","))

        total_true_s2 += sum(
            x.startswith("S2-")
            for x in ids
        )

        total_true_s3 += sum(
            x.startswith("S3-")
            for x in ids
        )


s2_recall = (
    found_true_s2 / total_true_s2
    if total_true_s2
    else 0
)

s3_recall = (
    found_true_s3 / total_true_s3
    if total_true_s3
    else 0
)

avg_candidates = (
    total_candidates / TEST_COUNT
)


print()
print("=" * 70)
print("BLOCKING RESULTS")
print("=" * 70)

print(f"True S2 matches: {total_true_s2:,}")
print(f"Found S2 matches: {found_true_s2:,}")
print(f"S2 recall: {s2_recall:.4f}")

print()

print(f"True S3 matches: {total_true_s3:,}")
print(f"Found S3 matches: {found_true_s3:,}")
print(f"S3 recall: {s3_recall:.4f}")

print()

print(
    f"Total candidate pairs: "
    f"{total_candidates:,}"
)

print(
    f"Average candidates per S1: "
    f"{avg_candidates:.2f}"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)