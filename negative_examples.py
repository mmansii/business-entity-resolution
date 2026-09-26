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
    sep="\t",
    nrows=20
)

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=200
)

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    nrows=200
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    nrows=20
)


# -----------------------------
# STORE REAL MATCHES
# -----------------------------

real_matches = set()

for _, row in ground_truth.iterrows():

    matched_ids = str(
        row["matched_entity_ids"]
    ).split(",")

    for match_id in matched_ids:

        if match_id and match_id != "nan":
            real_matches.add(match_id)


# -----------------------------
# CREATE NEGATIVE EXAMPLES
# -----------------------------

negative_examples = []


for _, s1 in source1.iterrows():

    s1_id = s1["entity_id"]

    s1_name = normalize_text(
        s1["business_name"]
    )

    s1_address = normalize_text(
        s1["business_address"]
    )

    s1_country = str(
        s1["country"]
    ).lower().strip()


    # Take some Source 2 businesses
    for _, row in source2.iterrows():

        match_id = row["entity_id"]

        # Skip if this is actually a known match
        if match_id in real_matches:
            continue

        name = normalize_text(
            row["business_name"]
        )

        address = normalize_text(
            row["business_address"]
        )

        country = str(
            row["country"]
        ).lower().strip()


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


        negative_examples.append({

            "source1_id": s1_id,

            "source2_or_3_id": match_id,

            "name_similarity": name_score,

            "address_similarity": address_score,

            "country_match": country_match,

            "label": 0

        })


# -----------------------------
# DATAFRAME
# -----------------------------

negative_df = pd.DataFrame(
    negative_examples
)


print("\nNEGATIVE EXAMPLES")
print("=" * 70)

print(
    negative_df.head(20).to_string(
        index=False
    )
)

print("\nNumber of negative examples:")

print(
    len(negative_df)
)


# -----------------------------
# SAVE
# -----------------------------

negative_df.to_csv(
    "negative_examples.csv",
    index=False
)

print("\nSaved as:")

print("negative_examples.csv")