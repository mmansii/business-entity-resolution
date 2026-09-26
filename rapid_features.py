import pandas as pd
import unicodedata
from rapidfuzz import fuzz


# --------------------------------------------------
# 1. Normalize text
# --------------------------------------------------

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Keep Hindi/other Unicode characters correctly
    text = unicodedata.normalize("NFC", text)

    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        # Keep letters, numbers, marks and spaces
        if category[0] in ("L", "N", "M") or char.isspace():
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    # Remove extra spaces
    text = " ".join(text.split())

    return text


# --------------------------------------------------
# 2. Similarity function
# --------------------------------------------------

def similarity(text1, text2):

    if not text1 or not text2:
        return 0.0

    return fuzz.ratio(text1, text2) / 100.0


# --------------------------------------------------
# 3. Calculate features for two businesses
# --------------------------------------------------

def calculate_features(row1, row2):

    name1 = normalize_text(row1["business_name"])
    name2 = normalize_text(row2["business_name"])

    address1 = normalize_text(row1["business_address"])
    address2 = normalize_text(row2["business_address"])

    country1 = normalize_text(row1["country"])
    country2 = normalize_text(row2["country"])

    # Name similarities
    name_similarity = similarity(name1, name2)

    name_token_similarity = 0.0

    if name1 and name2:
        name_token_similarity = (
            fuzz.token_set_ratio(name1, name2) / 100.0
        )

    # Address similarities
    address_similarity = similarity(address1, address2)

    address_token_similarity = 0.0

    if address1 and address2:
        address_token_similarity = (
            fuzz.token_set_ratio(address1, address2) / 100.0
        )

    # Whether addresses actually exist
    address1_present = int(bool(address1))
    address2_present = int(bool(address2))

    # Country match
    country_match = int(country1 == country2 and country1 != "")

    return {
        "name_similarity": name_similarity,
        "name_token_similarity": name_token_similarity,

        "address_similarity": address_similarity,
        "address_token_similarity": address_token_similarity,

        "address1_present": address1_present,
        "address2_present": address2_present,

        "country_match": country_match
    }


# --------------------------------------------------
# 4. Test the feature calculator
# --------------------------------------------------

if __name__ == "__main__":

    business1 = {
        "business_name": "Orelee's Barbershop",
        "business_address": "1795 Westchester Drive, High Point, NC",
        "country": "US"
    }

    business2 = {
        "business_name": "Orelee Barbershop",
        "business_address": "1795 Westchester Dr, High Point, NC",
        "country": "US"
    }

    features = calculate_features(business1, business2)

    print("\nFEATURES")
    print("=" * 50)

    for feature, value in features.items():
        print(f"{feature}: {value}")