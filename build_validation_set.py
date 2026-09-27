import pandas as pd
from rapidfuzz import fuzz
import unicodedata


# ============================================================
# 1. NORMALIZE TEXT
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
# 2. CALCULATE FEATURES
# ============================================================

def calculate_features(row1, row2):

    name1 = normalize_text(row1["business_name"])
    name2 = normalize_text(row2["business_name"])

    address1 = normalize_text(row1["business_address"])
    address2 = normalize_text(row2["business_address"])

    name_similarity = 0.0
    name_token_similarity = 0.0
    address_similarity = 0.0
    address_token_similarity = 0.0

    if name1 and name2:
        name_similarity = (
            fuzz.ratio(name1, name2) / 100
        )

        name_token_similarity = (
            fuzz.token_set_ratio(name1, name2) / 100
        )

    if address1 and address2:
        address_similarity = (
            fuzz.ratio(address1, address2) / 100
        )

        address_token_similarity = (
            fuzz.token_set_ratio(address1, address2) / 100
        )

    return {
        "name_similarity": name_similarity,
        "name_token_similarity": name_token_similarity,
        "address_similarity": address_similarity,
        "address_token_similarity": address_token_similarity,
        "address1_present": int(bool(address1)),
        "address2_present": int(bool(address2)),
        "country_match": int(
            normalize_text(row1["country"])
            == normalize_text(row2["country"])
        )
    }


# ============================================================
# 3. LOAD 100 SOURCE 1 RECORDS
# ============================================================

print("Loading Source 1...")

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=100
)

print(
    "Source 1 records loaded:",
    len(source1)
)


# ============================================================
# 4. LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

selected_source1_ids = set(
    source1["entity_id"]
)

ground_truth = ground_truth[
    ground_truth["source1_entity_id"].isin(
        selected_source1_ids
    )
].copy()

print(
    "Ground truth records for selected Source 1:",
    len(ground_truth)
)


# ============================================================
# 5. CREATE GROUND TRUTH LOOKUP
# ============================================================

ground_truth_map = {}

for _, row in ground_truth.iterrows():

    matched_ids = str(
        row["matched_entity_ids"]
    )

    if matched_ids == "nan":
        matched_ids = ""

    if matched_ids.strip():

        ids = set(
            matched_ids.split(",")
        )

    else:

        ids = set()

    ground_truth_map[
        row["source1_entity_id"]
    ] = ids


# ============================================================
# 6. COLLECT ALL GENUINE S2/S3 IDS
# ============================================================

genuine_s2_ids = set()
genuine_s3_ids = set()

for match_ids in ground_truth_map.values():

    for match_id in match_ids:

        if match_id.startswith("S2-"):
            genuine_s2_ids.add(match_id)

        elif match_id.startswith("S3-"):
            genuine_s3_ids.add(match_id)


print(
    "\nGenuine Source 2 IDs needed:",
    len(genuine_s2_ids)
)

print(
    "Genuine Source 3 IDs needed:",
    len(genuine_s3_ids)
)


# ============================================================
# 7. FIND EXACT GENUINE RECORDS IN SOURCE 2
# ============================================================

print("\nSearching Source 2 for genuine matches...")

genuine_source2 = []

for chunk in pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    chunksize=100000
):

    found = chunk[
        chunk["entity_id"].isin(genuine_s2_ids)
    ]

    if not found.empty:
        genuine_source2.append(found)

    if sum(
        len(x) for x in genuine_source2
    ) >= len(genuine_s2_ids):

        break


if genuine_source2:

    source2_genuine = pd.concat(
        genuine_source2,
        ignore_index=True
    )

else:

    source2_genuine = pd.DataFrame()


print(
    "Genuine Source 2 records found:",
    len(source2_genuine)
)


# ============================================================
# 8. FIND EXACT GENUINE RECORDS IN SOURCE 3
# ============================================================

print("\nSearching Source 3 for genuine matches...")

genuine_source3 = []

for chunk in pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    chunksize=100000
):

    found = chunk[
        chunk["entity_id"].isin(genuine_s3_ids)
    ]

    if not found.empty:
        genuine_source3.append(found)

    if sum(
        len(x) for x in genuine_source3
    ) >= len(genuine_s3_ids):

        break


if genuine_source3:

    source3_genuine = pd.concat(
        genuine_source3,
        ignore_index=True
    )

else:

    source3_genuine = pd.DataFrame()


print(
    "Genuine Source 3 records found:",
    len(source3_genuine)
)


# ============================================================
# 9. LOAD SMALL S2/S3 SAMPLES FOR HARD NEGATIVES
# ============================================================

print("\nLoading Source 2 sample for negatives...")

source2_sample = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=5000
)

print(
    "Source 2 negative sample:",
    len(source2_sample)
)


print("\nLoading Source 3 sample for negatives...")

source3_sample = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    nrows=5000
)

print(
    "Source 3 negative sample:",
    len(source3_sample)
)


# ============================================================
# 10. CREATE VALIDATION EXAMPLES
# ============================================================

examples = []


# ============================================================
# 11. CREATE GENUINE MATCHES
# ============================================================

print("\nCreating genuine matches...")

source2_lookup = {
    row["entity_id"]: row
    for _, row in source2_genuine.iterrows()
}

source3_lookup = {
    row["entity_id"]: row
    for _, row in source3_genuine.iterrows()
}


for _, row1 in source1.iterrows():

    source1_id = row1["entity_id"]

    true_matches = ground_truth_map.get(
        source1_id,
        set()
    )

    for match_id in true_matches:

        if match_id.startswith("S2-"):

            row2 = source2_lookup.get(
                match_id
            )

        elif match_id.startswith("S3-"):

            row2 = source3_lookup.get(
                match_id
            )

        else:

            continue

        if row2 is None:
            continue

        features = calculate_features(
            row1,
            row2
        )

        examples.append({

            "source1_id":
                source1_id,

            "matched_id":
                match_id,

            "source":
                match_id[:2],

            **features,

            "label":
                1
        })


# ============================================================
# 12. CREATE HARD NEGATIVES
# ============================================================

print("\nCreating hard negatives...")


def create_negatives(
    source,
    source_name
):

    for _, row1 in source1.iterrows():

        true_matches = ground_truth_map.get(
            row1["entity_id"],
            set()
        )

        same_country = source[
            source["country"]
            .astype(str)
            .str.lower()
            ==
            str(row1["country"]).lower()
        ]

        for _, row2 in same_country.iterrows():

            candidate_id = row2["entity_id"]

            if candidate_id in true_matches:
                continue

            features = calculate_features(
                row1,
                row2
            )

            if (
                features["name_similarity"] >= 0.50
                or
                features["name_token_similarity"] >= 0.50
                or
                features["address_similarity"] >= 0.50
                or
                features["address_token_similarity"] >= 0.50
            ):

                examples.append({

                    "source1_id":
                        row1["entity_id"],

                    "matched_id":
                        candidate_id,

                    "source":
                        source_name,

                    **features,

                    "label":
                        0
                })


# ============================================================
# 13. NEGATIVES FROM SOURCE 2
# ============================================================

print("Creating Source 2 hard negatives...")

create_negatives(
    source2_sample,
    "S2"
)


# ============================================================
# 14. NEGATIVES FROM SOURCE 3
# ============================================================

print("Creating Source 3 hard negatives...")

create_negatives(
    source3_sample,
    "S3"
)


# ============================================================
# 15. CREATE FINAL DATAFRAME
# ============================================================

validation_data = pd.DataFrame(
    examples
)


# ============================================================
# 16. SAVE
# ============================================================

validation_data.to_csv(
    "validation_data.csv",
    index=False
)


# ============================================================
# 17. SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATION DATASET CREATED")
print("=" * 70)

print(
    "\nTotal examples:",
    len(validation_data)
)

print(
    "\nLabels:"
)

print(
    validation_data["label"].value_counts()
)

print(
    "\nExamples by source and label:"
)

print(
    validation_data.groupby(
        ["source", "label"]
    ).size()
)

print(
    "\nSaved to validation_data.csv"
)