import pandas as pd
import re


print("=" * 70)
print("ANALYZING ADDRESS NUMBERS")
print("=" * 70)


# ============================================================
# LOAD MATCHING FEATURES
# ============================================================

df = pd.read_csv(
    "matching_features_v2_labeled.csv"
)


# ============================================================
# EXTRACT NUMBERS
# ============================================================

def extract_numbers(text):

    if pd.isna(text):
        return set()

    text = str(text).lower()

    numbers = re.findall(
        r"\d+[a-z]?",
        text
    )

    return set(numbers)


# ============================================================
# LOAD TRAINING SOURCE DATA
# IMPORTANT:
# matching_features_v2_labeled.csv was created from
# the training/validation data, so we need TRAIN files here.
# ============================================================

print()
print("Loading training source files...")


s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)


# ============================================================
# CREATE ADDRESS LOOKUP
# ============================================================

s1_addresses = dict(
    zip(
        s1["entity_id"],
        s1["business_address"]
    )
)

s2_addresses = dict(
    zip(
        s2["entity_id"],
        s2["business_address"]
    )
)

s3_addresses = dict(
    zip(
        s3["entity_id"],
        s3["business_address"]
    )
)


# ============================================================
# GET ADDRESS
# ============================================================

def get_address(entity_id):

    if entity_id.startswith("S1-"):

        return s1_addresses.get(
            entity_id,
            ""
        )

    if entity_id.startswith("S2-"):

        return s2_addresses.get(
            entity_id,
            ""
        )

    if entity_id.startswith("S3-"):

        return s3_addresses.get(
            entity_id,
            ""
        )

    return ""


# ============================================================
# CALCULATE NUMBER FEATURES
# ============================================================

print()
print("Calculating number features...")


number_match = []

number_overlap = []

number_count_1 = []

number_count_2 = []


for i, (_, row) in enumerate(
    df.iterrows()
):

    address1 = get_address(
        row["source1_id"]
    )

    address2 = get_address(
        row["matched_id"]
    )


    numbers1 = extract_numbers(
        address1
    )

    numbers2 = extract_numbers(
        address2
    )


    number_count_1.append(
        len(numbers1)
    )

    number_count_2.append(
        len(numbers2)
    )


    if not numbers1 or not numbers2:

        number_match.append(0)

        number_overlap.append(0.0)

    else:

        intersection = (
            numbers1 & numbers2
        )

        union = (
            numbers1 | numbers2
        )


        number_overlap.append(
            len(intersection)
            /
            len(union)
        )


        number_match.append(
            int(
                numbers1 == numbers2
            )
        )


    # Progress every 500,000 rows
    if (i + 1) % 500000 == 0:

        print(
            f"Processed {i + 1:,} rows..."
        )


# ============================================================
# ADD FEATURES
# ============================================================

df["number_match"] = number_match

df["number_overlap"] = number_overlap

df["number_count_1"] = number_count_1

df["number_count_2"] = number_count_2


# ============================================================
# POSITIVE / NEGATIVE
# ============================================================

positive = df[
    df["label"] == 1
]

negative = df[
    df["label"] == 0
]


print()
print("=" * 70)
print("NUMBER EXTRACTION CHECK")
print("=" * 70)

print(
    "Positive pairs with numbers:",
    (
        positive["number_overlap"] > 0
    ).sum()
)

print(
    "Negative pairs with numbers:",
    (
        negative["number_overlap"] > 0
    ).sum()
)


# ============================================================
# EXACT NUMBER SET MATCH
# ============================================================

print()
print("=" * 70)
print("EXACT NUMBER SET MATCH")
print("=" * 70)


for label, subset in [
    ("Positive", positive),
    ("Negative", negative)
]:

    count = (
        subset["number_match"] == 1
    ).sum()

    print(
        f"{label}: {count}"
    )


# ============================================================
# NUMBER OVERLAP
# ============================================================

print()
print("=" * 70)
print("NUMBER OVERLAP")
print("=" * 70)


for threshold in [
    0.25,
    0.50,
    0.75,
    1.00
]:

    pos = (
        positive["number_overlap"]
        >= threshold
    ).sum()

    neg = (
        negative["number_overlap"]
        >= threshold
    ).sum()


    print(
        f">= {threshold:.2f} "
        f"| Positive: {pos:5d} "
        f"| Negative: {neg:7d}"
    )


# ============================================================
# NUMBER + ADDRESS PRECISION
# ============================================================

print()
print("=" * 70)
print("NUMBER + ADDRESS PRECISION")
print("=" * 70)


conditions = [

    (
        "number overlap >= .50 + address sim >= .70",
        (
            (df["number_overlap"] >= 0.50)
            &
            (df["address_similarity"] >= 0.70)
        )
    ),

    (
        "number overlap >= .50 + address sim >= .80",
        (
            (df["number_overlap"] >= 0.50)
            &
            (df["address_similarity"] >= 0.80)
        )
    ),

    (
        "number overlap >= 1.00 + address sim >= .80",
        (
            (df["number_overlap"] >= 1.00)
            &
            (df["address_similarity"] >= 0.80)
        )
    ),

    (
        "exact number set + address sim >= .80",
        (
            (df["number_match"] == 1)
            &
            (df["address_similarity"] >= 0.80)
        )
    )
]


for description, condition in conditions:

    subset = df[
        condition
    ]

    tp = (
        subset["label"] == 1
    ).sum()

    fp = (
        subset["label"] == 0
    ).sum()


    precision = (
        tp / len(subset)
        if len(subset) > 0
        else 0
    )


    print()
    print(description)

    print(
        "Total:",
        len(subset)
    )

    print(
        "Genuine:",
        tp
    )

    print(
        "False:",
        fp
    )

    print(
        "Precision:",
        round(
            precision,
            4
        )
    )


# ============================================================
# FALSE POSITIVES WITH STRONG ADDRESS
# ============================================================

print()
print("=" * 70)
print("FALSE POSITIVES WITH STRONG ADDRESS")
print("=" * 70)


fp = df[
    (df["label"] == 0)
    &
    (df["address_similarity"] >= 0.80)
    &
    (df["address_token_similarity"] >= 0.90)
].copy()


print(
    "Count:",
    len(fp)
)


if len(fp) > 0:

    print()

    print(
        fp[
            [
                "source1_id",
                "matched_id",
                "source",
                "name_similarity",
                "name_token_similarity",
                "address_similarity",
                "address_token_similarity",
                "number_overlap",
                "number_match"
            ]
        ]
        .sort_values(
            [
                "number_overlap",
                "address_similarity"
            ],
            ascending=False
        )
        .head(50)
        .to_string(index=False)
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)