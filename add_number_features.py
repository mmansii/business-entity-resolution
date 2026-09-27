import pandas as pd
import re
import os

print("=" * 70)
print("ADDING ADDRESS NUMBER FEATURES")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load the existing matching features
# ---------------------------------------------------------

print("\nLoading matching features...")

features_file = "matching_features_v2_labeled.csv"

df = pd.read_csv(features_file)

print(f"Loaded {len(df):,} candidate pairs")

# ---------------------------------------------------------
# 2. Load the training source files
# ---------------------------------------------------------

print("\nLoading training source files...")

s1_file = "dataset/train/train_source1.tsv"
s2_file = "dataset/train/train_source2.tsv"
s3_file = "dataset/train/train_source3.tsv"

s1 = pd.read_csv(s1_file, sep="\t")
s2 = pd.read_csv(s2_file, sep="\t")
s3 = pd.read_csv(s3_file, sep="\t")

print(f"S1 rows: {len(s1):,}")
print(f"S2 rows: {len(s2):,}")
print(f"S3 rows: {len(s3):,}")

# ---------------------------------------------------------
# 3. Create address dictionaries
# ---------------------------------------------------------

print("\nCreating address dictionaries...")

s1_addresses = dict(
    zip(
        s1["entity_id"],
        s1["business_address"].fillna("").astype(str)
    )
)

s2_addresses = dict(
    zip(
        s2["entity_id"],
        s2["business_address"].fillna("").astype(str)
    )
)

s3_addresses = dict(
    zip(
        s3["entity_id"],
        s3["business_address"].fillna("").astype(str)
    )
)

# ---------------------------------------------------------
# 4. Extract numbers from addresses
# ---------------------------------------------------------

def extract_numbers(text):
    if not text:
        return set()

    text = str(text).lower()

    # Examples:
    # 123
    # 123a
    # 45b
    # 522a
    numbers = re.findall(r"\d+[a-z]?", text)

    return set(numbers)


def get_address(entity_id):

    if entity_id in s1_addresses:
        return s1_addresses[entity_id]

    if entity_id in s2_addresses:
        return s2_addresses[entity_id]

    if entity_id in s3_addresses:
        return s3_addresses[entity_id]

    return ""


# ---------------------------------------------------------
# 5. Calculate number features
# ---------------------------------------------------------

print("\nCalculating number features...")

number_overlaps = []
number_matches = []

total = len(df)

for i, row in df.iterrows():

    address1 = get_address(row["source1_id"])
    address2 = get_address(row["matched_id"])

    nums1 = extract_numbers(address1)
    nums2 = extract_numbers(address2)

    # No numbers in either address
    if not nums1 or not nums2:
        overlap = 0.0
        exact_match = 0

    else:
        common = nums1.intersection(nums2)

        # overlap relative to the smaller set
        overlap = len(common) / min(len(nums1), len(nums2))

        exact_match = int(nums1 == nums2)

    number_overlaps.append(overlap)
    number_matches.append(exact_match)

    if (i + 1) % 500000 == 0:
        print(f"Processed {i + 1:,} rows...")


df["number_overlap"] = number_overlaps
df["number_match"] = number_matches

# ---------------------------------------------------------
# 6. Save result
# ---------------------------------------------------------

output_file = "matching_features_v2_labeled_numbers.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"\nSaved:")
print(output_file)

print(f"\nRows: {len(df):,}")

print("\nNew columns:")
print("  number_overlap")
print("  number_match")

print("\nNumber match count:")
print(df["number_match"].sum())

print("\nNumber overlap >= 0.50:")
print((df["number_overlap"] >= 0.50).sum())

print("\nNumber overlap >= 0.75:")
print((df["number_overlap"] >= 0.75).sum())

print("\nNumber overlap = 1.00:")
print((df["number_overlap"] >= 1.00).sum())