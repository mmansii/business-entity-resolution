import unicodedata
import pandas as pd


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Normalize Unicode representation
    text = unicodedata.normalize("NFC", text)

    # Keep letters, numbers, marks and spaces
    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if category[0] in ("L", "N", "M") or char.isspace():
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    # Remove extra spaces
    text = " ".join(text.split())

    return text


df = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=5
)

df["name_normalized"] = df["business_name"].apply(normalize_text)
df["address_normalized"] = df["business_address"].apply(normalize_text)

print(df[[
    "business_name",
    "name_normalized",
    "business_address",
    "address_normalized"
]])