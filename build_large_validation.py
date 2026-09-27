import pandas as pd
import unicodedata
from rapidfuzz import fuzz


# =========================================================
# 1. SETTINGS
# =========================================================

NUM_SOURCE1 = 1000
NEGATIVE_POOL_SIZE = 10000

print("=" * 70)
print("BUILDING LARGE VALIDATION DATASET")
print("=" * 70)


# =========================================================
# 2. NORMALIZATION
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
# 3. SIMILARITY
# =========================================================

def similarity(text1, text2):

    if not text1 or not text2:
        return 0.0

    return fuzz.ratio(text1, text2) / 100.0


# =========================================================
# 4. FEATURE CALCULATION
# =========================================================

def calculate_features(row1, row2):

    name1 = normalize_text(row1["business_name"])
    name2 = normalize_text(row2["business_name"])

    address1 = normalize_text(row1["business_address"])
    address2 = normalize_text(row2["business_address"])

    country1 = normalize_text(row1["country"])
    country2 = normalize_text(row2["country"])

    name_similarity = similarity(name1, name2)

    name_token_similarity = 0.0

    if name1 and name2:
        name_token_similarity = (
            fuzz.token_set_ratio(name1, name2) / 100.0
        )

    address_similarity = similarity(address1, address2)

    address_token_similarity = 0.0

    if address1 and address2:
        address_token_similarity = (
            fuzz.token_set_ratio(address1, address2) / 100.0
        )

    address1_present = int(bool(address1))
    address2_present = int(bool(address2))

    country_match = int(
        country1 == country2 and country1 != ""
    )

    return {
        "name_similarity": name_similarity,
        "name_token_similarity": name_token_similarity,
        "address_similarity": address_similarity,
        "address_token_similarity": address_token_similarity,
        "address1_present": address1_present,
        "address2_present": address2_present,
        "country_match": country_match
    }


# =========================================================
# 5. LOAD SOURCE 1
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
# 6. LOAD GROUND TRUTH
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
# 7. LOAD SOURCE 2 AND SOURCE 3
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
# 8. CREATE LOOKUP DICTIONARY
# =========================================================

print()
print("Creating entity lookup...")

source2_lookup = source2.set_index("entity_id").to_dict("index")
source3_lookup = source3.set_index("entity_id").to_dict("index")

print("Lookup ready.")


# =========================================================
# 9. CREATE POSITIVE PAIRS
# =========================================================

print()
print("Creating genuine positive pairs...")

positive_pairs = []

for _, gt_row in ground_truth.iterrows():

    source1_id = gt_row["source1_entity_id"]

    matched_ids = gt_row["matched_entity_ids"]

    if pd.isna(matched_ids):
        continue

    matched_ids = str(matched_ids).strip()

    if not matched_ids:
        continue

    for matched_id in matched_ids.split(","):

        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):

            row2 = source2_lookup.get(matched_id)

            source_name = "S2"

        elif matched_id.startswith("S3-"):

            row2 = source3_lookup.get(matched_id)

            source_name = "S3"

        else:
            continue

        if row2 is None:
            continue

        row1 = source1[
            source1["entity_id"] == source1_id
        ]

        if row1.empty:
            continue

        row1 = row1.iloc[0]

        features = calculate_features(row1, row2)

        positive_pairs.append({
            "source1_id": source1_id,
            "matched_id": matched_id,
            "source": source_name,
            **features,
            "label": 1
        })


print("Positive pairs:", len(positive_pairs))


# =========================================================
# 10. CREATE NEGATIVE PAIRS
# =========================================================

print()
print("Creating negative pairs...")

# Combine Source 2 and Source 3
all_candidates = pd.concat(
    [
        source2,
        source3
    ],
    ignore_index=True
)

# Take a fixed pool so the experiment remains manageable
all_candidates = all_candidates.head(
    NEGATIVE_POOL_SIZE
).copy()

# Ground-truth matches for each Source 1
true_matches = {}

for _, gt_row in ground_truth.iterrows():

    source1_id = gt_row["source1_entity_id"]

    matched_ids = gt_row["matched_entity_ids"]

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


negative_pairs = []

for _, row1 in source1.iterrows():

    source1_id = row1["entity_id"]

    known_matches = true_matches.get(
        source1_id,
        set()
    )

    # Only compare candidates from the same country.
    same_country = all_candidates[
        all_candidates["country"].fillna("").str.lower()
        ==
        str(row1["country"]).lower()
    ]

    for _, row2 in same_country.iterrows():

        matched_id = row2["entity_id"]

        # Never use a genuine match as a negative.
        if matched_id in known_matches:
            continue

        features = calculate_features(
            row1,
            row2
        )

        negative_pairs.append({
            "source1_id": source1_id,
            "matched_id": matched_id,
            "source": (
                "S2"
                if matched_id.startswith("S2-")
                else "S3"
            ),
            **features,
            "label": 0
        })


print("Negative pairs:", len(negative_pairs))


# =========================================================
# 11. COMBINE
# =========================================================

print()
print("Combining positive and negative pairs...")

validation = pd.DataFrame(
    positive_pairs + negative_pairs
)

print("Total validation pairs:", len(validation))


# =========================================================
# 12. SAVE
# =========================================================

validation.to_csv(
    "validation_large.csv",
    index=False
)

print()
print("=" * 70)
print("DONE!")
print("=" * 70)

print("Saved:")
print("validation_large.csv")

print()
print("Label counts:")
print(validation["label"].value_counts())

print()
print("Source counts:")
print(validation["source"].value_counts())

print()
print("=" * 70)