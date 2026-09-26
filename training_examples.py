import pandas as pd
import unicodedata
from difflib import SequenceMatcher


# -----------------------------
# NORMALIZATION
# -----------------------------

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


# -----------------------------
# SIMILARITY
# -----------------------------

def similarity(text1, text2):

    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


# -----------------------------
# LOAD DATA
# -----------------------------

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    nrows=20
)


# -----------------------------
# CREATE POSITIVE EXAMPLES
# -----------------------------

training_examples = []

for _, gt_row in ground_truth.iterrows():

    source1_id = gt_row["source1_entity_id"]

    s1_result = source1[
        source1["entity_id"] == source1_id
    ]

    if len(s1_result) == 0:
        continue

    s1 = s1_result.iloc[0]

    s1_name = normalize_text(s1["business_name"])
    s1_address = normalize_text(s1["business_address"])
    s1_country = str(s1["country"]).lower().strip()

    matched_ids = str(
        gt_row["matched_entity_ids"]
    ).split(",")

    for match_id in matched_ids:

        if not match_id or match_id == "nan":
            continue

        # Find matching record
        if match_id.startswith("S2-"):

            result = source2[
                source2["entity_id"] == match_id
            ]

        elif match_id.startswith("S3-"):

            result = source3[
                source3["entity_id"] == match_id
            ]

        else:
            continue

        if len(result) == 0:
            continue

        row = result.iloc[0]

        name = normalize_text(row["business_name"])
        address = normalize_text(row["business_address"])
        country = str(row["country"]).lower().strip()

        name_score = similarity(
            s1_name,
            name
        )

        address_score = similarity(
            s1_address,
            address
        )

        country_match = int(
            s1_country == country
        )

        training_examples.append({
            "source1_id": source1_id,
            "source2_or_3_id": match_id,
            "name_similarity": name_score,
            "address_similarity": address_score,
            "country_match": country_match,
            "label": 1
        })


# -----------------------------
# CREATE DATAFRAME
# -----------------------------

training_df = pd.DataFrame(training_examples)


print("\nTRAINING EXAMPLES")
print("=" * 70)

print(training_df.head(20).to_string(index=False))

print("\nNumber of positive examples:")
print(len(training_df))


# -----------------------------
# SAVE
# -----------------------------

training_df.to_csv(
    "positive_examples.csv",
    index=False
)

print("\nSaved as:")
print("positive_examples.csv")