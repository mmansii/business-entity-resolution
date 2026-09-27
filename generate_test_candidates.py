import pandas as pd
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
import time


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = Path("dataset/test")
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(exist_ok=True)

S1_FILE = TEST_DIR / "test_source1.tsv"
S2_FILE = TEST_DIR / "test_source2.tsv"
S3_FILE = TEST_DIR / "test_source3.tsv"

OUTPUT_FILE = OUTPUT_DIR / "test_candidates.tsv"


# ============================================================
# TEXT NORMALIZATION
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
    text = " ".join(text.split())

    return text


# ============================================================
# EXTRACT NUMBERS
# ============================================================

def extract_numbers(text):
    if not text:
        return []

    return re.findall(r"\d+[a-z]?", text)


# ============================================================
# MEANINGFUL TOKENS
# ============================================================

STOPWORDS = {
    "the",
    "and",
    "of",
    "for",
    "inc",
    "llc",
    "ltd",
    "limited",
    "company",
    "co",
    "corp",
    "corporation",
    "private",
    "pvt",
    "plc",
    "llp",
    "india",
    "ind",
    "services",
    "service",
    "business",
    "group",
    "center",
    "centre",
    "international",
}


def meaningful_tokens(text):
    tokens = text.split()

    result = []

    for token in tokens:

        if len(token) < 3:
            continue

        if token in STOPWORDS:
            continue

        result.append(token)

    return result


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

start = time.time()

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

print(f"S1 rows: {len(s1):,}")
print(f"S2 rows: {len(s2):,}")
print(f"S3 rows: {len(s3):,}")

print(f"Loading time: {time.time() - start:.1f} seconds")


# ============================================================
# NORMALIZE
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING TEXT")
print("=" * 70)

start = time.time()

for df in [s1, s2, s3]:

    df["name_norm"] = df["business_name"].map(normalize_text)

    df["address_norm"] = df["business_address"].map(normalize_text)

    df["country_norm"] = df["country"].map(normalize_text)

print(f"Normalization time: {time.time() - start:.1f} seconds")


# ============================================================
# BUILD BLOCK INDEX
# ============================================================

def build_index(df):

    token_index = defaultdict(list)
    address_index = defaultdict(list)
    prefix_index = defaultdict(list)
    number_index = defaultdict(list)

    for idx, row in enumerate(df.itertuples(index=False)):

        country = row.country_norm
        name = row.name_norm
        address = row.address_norm

        if not country:
            continue

        # ----------------------------------------------------
        # NAME TOKEN BLOCK
        # ----------------------------------------------------

        tokens = meaningful_tokens(name)

        for token in set(tokens):

            key = (country, token)

            token_index[key].append(idx)

        # ----------------------------------------------------
        # ADDRESS TOKEN BLOCK
        # ----------------------------------------------------

        address_tokens = meaningful_tokens(address)

        for token in set(address_tokens):

            key = (country, token)

            address_index[key].append(idx)

        # ----------------------------------------------------
        # NAME PREFIX BLOCK
        # ----------------------------------------------------

        if name:

            compact_name = name.replace(" ", "")

            if len(compact_name) >= 3:

                prefix = compact_name[:3]

                key = (country, prefix)

                prefix_index[key].append(idx)

        # ----------------------------------------------------
        # ADDRESS NUMBER BLOCK
        # ----------------------------------------------------

        numbers = extract_numbers(address)

        for number in set(numbers):

            key = (country, number)

            number_index[key].append(idx)

    return (
        token_index,
        address_index,
        prefix_index,
        number_index
    )


print("\n" + "=" * 70)
print("BUILDING S2 INDEX")
print("=" * 70)

start = time.time()

s2_token, s2_address, s2_prefix, s2_number = build_index(s2)

print(f"S2 index time: {time.time() - start:.1f} seconds")


print("\n" + "=" * 70)
print("BUILDING S3 INDEX")
print("=" * 70)

start = time.time()

s3_token, s3_address, s3_prefix, s3_number = build_index(s3)

print(f"S3 index time: {time.time() - start:.1f} seconds")


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(row, token_index, address_index,
                        prefix_index, number_index):

    country = row.country_norm
    name = row.name_norm
    address = row.address_norm

    candidates = set()

    # --------------------------------------------------------
    # NAME TOKEN BLOCK
    # --------------------------------------------------------

    for token in set(meaningful_tokens(name)):

        key = (country, token)

        candidates.update(token_index.get(key, []))

    # --------------------------------------------------------
    # ADDRESS TOKEN BLOCK
    # --------------------------------------------------------

    for token in set(meaningful_tokens(address)):

        key = (country, token)

        candidates.update(address_index.get(key, []))

    # --------------------------------------------------------
    # NAME PREFIX BLOCK
    # --------------------------------------------------------

    if name:

        compact_name = name.replace(" ", "")

        if len(compact_name) >= 3:

            prefix = compact_name[:3]

            key = (country, prefix)

            candidates.update(prefix_index.get(key, []))

    # --------------------------------------------------------
    # ADDRESS NUMBER BLOCK
    # --------------------------------------------------------

    for number in set(extract_numbers(address)):

        key = (country, number)

        candidates.update(number_index.get(key, []))

    return candidates


# ============================================================
# GENERATE AND WRITE CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TEST CANDIDATES")
print("=" * 70)

start = time.time()

total_s2 = 0
total_s3 = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write("source1_id\tmatched_id\tsource\n")

    for counter, row in enumerate(s1.itertuples(index=False), start=1):

        candidates_s2 = generate_candidates(
            row,
            s2_token,
            s2_address,
            s2_prefix,
            s2_number
        )

        candidates_s3 = generate_candidates(
            row,
            s3_token,
            s3_address,
            s3_prefix,
            s3_number
        )

        source1_id = row.entity_id

        for idx in candidates_s2:

            f.write(
                f"{source1_id}\t{s2.iloc[idx].entity_id}\tS2\n"
            )

        for idx in candidates_s3:

            f.write(
                f"{source1_id}\t{s3.iloc[idx].entity_id}\tS3\n"
            )

        total_s2 += len(candidates_s2)
        total_s3 += len(candidates_s3)

        if counter % 10000 == 0:

            elapsed = time.time() - start

            print(
                f"Processed {counter:,}/{len(s1):,} "
                f"| S2 candidates: {total_s2:,} "
                f"| S3 candidates: {total_s3:,} "
                f"| Time: {elapsed/60:.1f} min"
            )


# ============================================================
# SUMMARY
# ============================================================

elapsed = time.time() - start

print("\n" + "=" * 70)
print("CANDIDATE GENERATION COMPLETE")
print("=" * 70)

print(f"S1 entities: {len(s1):,}")

print(f"S2 candidate pairs: {total_s2:,}")
print(f"S3 candidate pairs: {total_s3:,}")

print(
    f"Total candidate pairs: "
    f"{total_s2 + total_s3:,}"
)

print(
    f"Average candidates per S1: "
    f"{(total_s2 + total_s3) / len(s1):.2f}"
)

print(f"Runtime: {elapsed / 60:.1f} minutes")

print(f"\nSaved to:")
print(OUTPUT_FILE)