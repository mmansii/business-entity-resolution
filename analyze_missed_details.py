import pandas as pd
import unicodedata


print("=" * 70)
print("ANALYZING MISSED BLOCKING MATCHES")
print("=" * 70)


# =========================================================
# LOAD SOURCE 1
# =========================================================

print()
print("Loading Source 1...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)


# =========================================================
# LOAD SOURCE 2
# =========================================================

print("Loading Source 2...")

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)


# =========================================================
# LOAD SOURCE 3
# =========================================================

print("Loading Source 3...")

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)


# =========================================================
# LOAD MISSES
# =========================================================

print("Loading blocking misses...")

misses = pd.read_csv(
    "blocking_misses.csv"
)


# =========================================================
# NORMALIZATION
# =========================================================

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


# =========================================================
# CREATE LOOKUPS
# =========================================================

print()
print("Creating lookups...")

s1_lookup = source1.set_index(
    "entity_id"
).to_dict("index")

s2_lookup = source2.set_index(
    "entity_id"
).to_dict("index")

s3_lookup = source3.set_index(
    "entity_id"
).to_dict("index")


# =========================================================
# ANALYZE
# =========================================================

results = []


for _, miss in misses.iterrows():

    s1_id = miss["source1_id"]
    matched_id = miss["matched_id"]

    row1 = s1_lookup.get(s1_id)

    if matched_id.startswith("S2-"):
        row2 = s2_lookup.get(matched_id)
        source = "S2"
    else:
        row2 = s3_lookup.get(matched_id)
        source = "S3"

    if row1 is None or row2 is None:
        continue

    name1 = normalize_text(
        row1["business_name"]
    )

    name2 = normalize_text(
        row2["business_name"]
    )

    address1 = normalize_text(
        row1["business_address"]
    )

    address2 = normalize_text(
        row2["business_address"]
    )

    # Tokens
    name_tokens1 = set(
        x for x in name1.split()
        if len(x) >= 3
    )

    name_tokens2 = set(
        x for x in name2.split()
        if len(x) >= 3
    )

    address_tokens1 = set(
        x for x in address1.split()
        if len(x) >= 3
    )

    address_tokens2 = set(
        x for x in address2.split()
        if len(x) >= 3
    )

    common_name_tokens = (
        name_tokens1 &
        name_tokens2
    )

    common_address_tokens = (
        address_tokens1 &
        address_tokens2
    )

    results.append({

        "source1_id": s1_id,

        "matched_id": matched_id,

        "source": source,

        "name1": row1["business_name"],

        "name2": row2["business_name"],

        "address1": row1["business_address"],

        "address2": row2["business_address"],

        "country": row1["country"],

        "normalized_name1": name1,

        "normalized_name2": name2,

        "normalized_address1": address1,

        "normalized_address2": address2,

        "common_name_tokens": ", ".join(
            sorted(common_name_tokens)
        ),

        "common_address_tokens": ", ".join(
            sorted(common_address_tokens)
        ),

        "name_token_count": len(
            common_name_tokens
        ),

        "address_token_count": len(
            common_address_tokens
        ),

        "name_prefix1": name1[:3],

        "name_prefix2": name2[:3],

        "address_prefix1": address1[:5],

        "address_prefix2": address2[:5]
    })


# =========================================================
# SAVE
# =========================================================

result = pd.DataFrame(results)

result.to_csv(
    "blocking_miss_details.csv",
    index=False
)


# =========================================================
# PRINT RESULTS
# =========================================================

print()
print("=" * 70)
print("MISSED MATCH DETAILS")
print("=" * 70)

for _, row in result.iterrows():

    print()
    print("-" * 70)

    print(
        row["source1_id"],
        "->",
        row["matched_id"],
        "(",
        row["source"],
        ")"
    )

    print(
        "S1 Name:",
        row["name1"]
    )

    print(
        "Matched Name:",
        row["name2"]
    )

    print(
        "S1 Address:",
        row["address1"]
    )

    print(
        "Matched Address:",
        row["address2"]
    )

    print(
        "Common name tokens:",
        row["common_name_tokens"]
    )

    print(
        "Common address tokens:",
        row["common_address_tokens"]
    )

    print(
        "Name prefixes:",
        row["name_prefix1"],
        "|",
        row["name_prefix2"]
    )

    print(
        "Address prefixes:",
        row["address_prefix1"],
        "|",
        row["address_prefix2"]
    )


print()
print("=" * 70)
print("Saved:")
print("blocking_miss_details.csv")
print("=" * 70)