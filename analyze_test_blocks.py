import pandas as pd
import re
import unicodedata
from collections import Counter
from pathlib import Path
import time
import gc


TEST_DIR = Path("dataset/test")
CHUNK_SIZE = 100_000


STOPWORDS = {
    "the", "and", "of", "for", "inc", "llc", "ltd",
    "limited", "company", "co", "corp", "corporation",
    "private", "pvt", "plc", "llp", "india", "ind",
    "services", "service", "business", "group",
    "center", "centre", "international"
}


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
    result = []

    for token in text.split():

        if len(token) < 3:
            continue

        if token in STOPWORDS:
            continue

        result.append(token)

    return result


def extract_numbers(text):
    if not text:
        return []

    return re.findall(r"\d+[a-z]?", text)


# ============================================================
# COUNTERS
# ============================================================

name_token_counts = Counter()
address_token_counts = Counter()
prefix_counts = Counter()
number_counts = Counter()


# ============================================================
# PROCESS S2 + S3
# ============================================================

files = [
    ("S2", TEST_DIR / "test_source2.tsv"),
    ("S3", TEST_DIR / "test_source3.tsv")
]

start = time.time()

for source_name, file_path in files:

    print()
    print("=" * 70)
    print(f"ANALYZING {source_name}")
    print("=" * 70)

    processed = 0

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        usecols=[
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        chunk = chunk.fillna("")

        for row in chunk.itertuples(index=False):

            country = normalize_text(row.country)

            if not country:
                continue

            name = normalize_text(row.business_name)
            address = normalize_text(row.business_address)

            # ------------------------------------------------
            # NAME TOKENS
            # ------------------------------------------------

            for token in set(meaningful_tokens(name)):

                name_token_counts[
                    (country, token)
                ] += 1

            # ------------------------------------------------
            # ADDRESS TOKENS
            # ------------------------------------------------

            for token in set(meaningful_tokens(address)):

                address_token_counts[
                    (country, token)
                ] += 1

            # ------------------------------------------------
            # NAME PREFIX
            # ------------------------------------------------

            compact_name = name.replace(" ", "")

            if len(compact_name) >= 3:

                prefix_counts[
                    (country, compact_name[:3])
                ] += 1

            # ------------------------------------------------
            # ADDRESS NUMBERS
            # ------------------------------------------------

            for number in set(extract_numbers(address)):

                number_counts[
                    (country, number)
                ] += 1

            processed += 1

        if processed % 500_000 < CHUNK_SIZE:

            print(
                f"Processed {processed:,} rows..."
            )

        del chunk
        gc.collect()


# ============================================================
# STATISTICS
# ============================================================

def print_stats(name, counter):

    values = list(counter.values())

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Unique blocks: {len(values):,}")

    if not values:
        return

    values.sort()

    print(f"Smallest: {values[0]:,}")
    print(f"Median: {values[len(values)//2]:,}")
    print(f"90th percentile: {values[int(len(values)*0.90)]:,}")
    print(f"95th percentile: {values[int(len(values)*0.95)]:,}")
    print(f"99th percentile: {values[int(len(values)*0.99)]:,}")
    print(f"Maximum: {values[-1]:,}")

    for limit in [1, 5, 10, 25, 50, 100, 250, 500, 1000]:

        count = sum(v <= limit for v in values)

        print(
            f"Blocks with <= {limit:4}: "
            f"{count:,} "
            f"({count / len(values) * 100:.2f}%)"
        )


print_stats(
    "NAME TOKEN BLOCKS",
    name_token_counts
)

print_stats(
    "ADDRESS TOKEN BLOCKS",
    address_token_counts
)

print_stats(
    "NAME PREFIX BLOCKS",
    prefix_counts
)

print_stats(
    "ADDRESS NUMBER BLOCKS",
    number_counts
)


print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Total runtime: "
    f"{(time.time() - start) / 60:.1f} minutes"
)