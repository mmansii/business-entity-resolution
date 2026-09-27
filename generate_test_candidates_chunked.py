import pandas as pd
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
import time
import gc


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

CHUNK_SIZE = 100_000


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
# NUMBER EXTRACTION
# ============================================================

def extract_numbers(text):

    if not text:
        return []

    return re.findall(r"\d+[a-z]?", text)


# ============================================================
# TOKENS
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

    result = []

    for token in text.split():

        if len(token) < 3:
            continue

        if token in STOPWORDS:
            continue

        result.append(token)

    return result


# ============================================================
# BUILD INDEX FROM SOURCE 2 / SOURCE 3
# ============================================================

def build_index(file_path, source_name):

    print()
    print("=" * 70)
    print(f"BUILDING {source_name} INDEX")
    print("=" * 70)

    start = time.time()

    token_index = defaultdict(list)
    address_index = defaultdict(list)
    prefix_index = defaultdict(list)
    number_index = defaultdict(list)

    row_count = 0

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        chunk = chunk.fillna("")

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id
            name = normalize_text(row.business_name)
            address = normalize_text(row.business_address)
            country = normalize_text(row.country)

            if not country:
                row_count += 1
                continue

            # ------------------------------------------------
            # NAME TOKEN BLOCK
            # ------------------------------------------------

            for token in set(meaningful_tokens(name)):

                key = (country, token)

                token_index[key].append(entity_id)

            # ------------------------------------------------
            # ADDRESS TOKEN BLOCK
            # ------------------------------------------------

            for token in set(meaningful_tokens(address)):

                key = (country, token)

                address_index[key].append(entity_id)

            # ------------------------------------------------
            # NAME PREFIX BLOCK
            # ------------------------------------------------

            compact_name = name.replace(" ", "")

            if len(compact_name) >= 3:

                prefix = compact_name[:3]

                key = (country, prefix)

                prefix_index[key].append(entity_id)

            # ------------------------------------------------
            # ADDRESS NUMBER BLOCK
            # ------------------------------------------------

            for number in set(extract_numbers(address)):

                key = (country, number)

                number_index[key].append(entity_id)

            row_count += 1

        if row_count % 500_000 < CHUNK_SIZE:

            print(
                f"Processed {row_count:,} {source_name} rows..."
            )

        del chunk
        gc.collect()

    elapsed = time.time() - start

    print(f"{source_name} rows indexed: {row_count:,}")
    print(f"Index time: {elapsed / 60:.1f} minutes")

    return (
        token_index,
        address_index,
        prefix_index,
        number_index
    )


# ============================================================
# GENERATE CANDIDATES FOR ONE S1 ROW
# ============================================================

def generate_candidates(
    row,
    token_index,
    address_index,
    prefix_index,
    number_index
):

    country = normalize_text(row.country)
    name = normalize_text(row.business_name)
    address = normalize_text(row.business_address)

    candidates = set()

    # --------------------------------------------------------
    # NAME TOKEN
    # --------------------------------------------------------

    for token in set(meaningful_tokens(name)):

        key = (country, token)

        candidates.update(
            token_index.get(key, [])
        )

    # --------------------------------------------------------
    # ADDRESS TOKEN
    # --------------------------------------------------------

    for token in set(meaningful_tokens(address)):

        key = (country, token)

        candidates.update(
            address_index.get(key, [])
        )

    # --------------------------------------------------------
    # NAME PREFIX
    # --------------------------------------------------------

    compact_name = name.replace(" ", "")

    if len(compact_name) >= 3:

        prefix = compact_name[:3]

        key = (country, prefix)

        candidates.update(
            prefix_index.get(key, [])
        )

    # --------------------------------------------------------
    # ADDRESS NUMBER
    # --------------------------------------------------------

    for number in set(extract_numbers(address)):

        key = (country, number)

        candidates.update(
            number_index.get(key, [])
        )

    return candidates


# ============================================================
# BUILD S2 INDEX
# ============================================================

s2_indexes = build_index(
    S2_FILE,
    "S2"
)

gc.collect()


# ============================================================
# BUILD S3 INDEX
# ============================================================

s3_indexes = build_index(
    S3_FILE,
    "S3"
)

gc.collect()


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================

if OUTPUT_FILE.exists():

    print()
    print("Removing previous candidate file...")

    OUTPUT_FILE.unlink()


# ============================================================
# PROCESS S1 IN CHUNKS
# ============================================================

print()
print("=" * 70)
print("GENERATING TEST CANDIDATES")
print("=" * 70)

start = time.time()

total_s2 = 0
total_s3 = 0
processed = 0


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as output:

    output.write(
        "source1_id\tmatched_id\tsource\n"
    )

    for chunk in pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        chunk = chunk.fillna("")

        for row in chunk.itertuples(index=False):

            source1_id = row.entity_id

            # ------------------------------------------------
            # S2
            # ------------------------------------------------

            candidates_s2 = generate_candidates(
                row,
                *s2_indexes
            )

            for matched_id in candidates_s2:

                output.write(
                    f"{source1_id}\t{matched_id}\tS2\n"
                )

            # ------------------------------------------------
            # S3
            # ------------------------------------------------

            candidates_s3 = generate_candidates(
                row,
                *s3_indexes
            )

            for matched_id in candidates_s3:

                output.write(
                    f"{source1_id}\t{matched_id}\tS3\n"
                )

            total_s2 += len(candidates_s2)
            total_s3 += len(candidates_s3)

            processed += 1

        del chunk
        gc.collect()

        if processed % 500_000 < CHUNK_SIZE:

            elapsed = time.time() - start

            print(
                f"Processed S1: {processed:,} / 1,732,544 "
                f"| S2 candidates: {total_s2:,} "
                f"| S3 candidates: {total_s3:,} "
                f"| Time: {elapsed / 60:.1f} min"
            )


# ============================================================
# SUMMARY
# ============================================================

elapsed = time.time() - start

print()
print("=" * 70)
print("CANDIDATE GENERATION COMPLETE")
print("=" * 70)

print(f"S1 entities: {processed:,}")

print(f"S2 candidate pairs: {total_s2:,}")
print(f"S3 candidate pairs: {total_s3:,}")

print(
    f"Total candidate pairs: "
    f"{total_s2 + total_s3:,}"
)

print(
    f"Average candidates per S1: "
    f"{(total_s2 + total_s3) / processed:.2f}"
)

print(
    f"Runtime: {elapsed / 60:.1f} minutes"
)

print()
print(f"Output: {OUTPUT_FILE}")