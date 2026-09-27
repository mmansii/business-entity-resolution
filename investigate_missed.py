import pandas as pd
import unicodedata


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


# ---------------------------------------------------------
# Load Source 1
# ---------------------------------------------------------

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

# ---------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

gt = dict(
    zip(
        ground_truth["source1_entity_id"],
        ground_truth["matched_entity_ids"]
    )
)


# ---------------------------------------------------------
# Get S1-800482427
# ---------------------------------------------------------

s1_id = "S1-800482427"

s1_row = source1[
    source1["entity_id"] == s1_id
].iloc[0]

print("\nSOURCE 1 RECORD")
print("=" * 70)

print("ID:", s1_row["entity_id"])
print("Name:", s1_row["business_name"])
print("Address:", s1_row["business_address"])
print("Country:", s1_row["country"])


# ---------------------------------------------------------
# Get its genuine matches
# ---------------------------------------------------------

matches = gt[s1_id]

print("\nGROUND TRUTH MATCHES")
print("=" * 70)

print(matches)


true_s3_ids = [
    x.strip()
    for x in str(matches).split(",")
    if x.startswith("S3-")
]


print("\nS3 MATCHES")
print("=" * 70)

print(true_s3_ids)


# ---------------------------------------------------------
# Search Source 3
# ---------------------------------------------------------

found_rows = []

for chunk in pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t",
    chunksize=100000
):

    found = chunk[
        chunk["entity_id"].isin(true_s3_ids)
    ]

    if len(found) > 0:
        found_rows.append(found)


true_rows = pd.concat(
    found_rows,
    ignore_index=True
)


# ---------------------------------------------------------
# Display genuine S3 records
# ---------------------------------------------------------

print("\nGENUINE S3 RECORD DETAILS")
print("=" * 70)

for _, row in true_rows.iterrows():

    print("\nID:", row["entity_id"])
    print("Name:", row["business_name"])
    print("Address:", row["business_address"])
    print("Country:", row["country"])


# ---------------------------------------------------------
# Show normalized versions
# ---------------------------------------------------------

print("\nNORMALIZED VALUES")
print("=" * 70)

print(
    "\nS1 name:",
    normalize_text(s1_row["business_name"])
)

print(
    "S1 address:",
    normalize_text(s1_row["business_address"])
)

for _, row in true_rows.iterrows():

    print("\n", row["entity_id"])

    print(
        "Name:",
        normalize_text(row["business_name"])
    )

    print(
        "Address:",
        normalize_text(row["business_address"])
    )