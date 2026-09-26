import pandas as pd
from difflib import SequenceMatcher


def similarity(text1, text2):
    if pd.isna(text1) or pd.isna(text2):
        return 0.0

    text1 = str(text1).lower()
    text2 = str(text2).lower()

    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


def calculate_features(row1, row2):

    name_score = similarity(
        row1["name_normalized"],
        row2["name_normalized"]
    )

    address_score = similarity(
        row1["address_normalized"],
        row2["address_normalized"]
    )

    country_match = int(
        row1["country"].lower().strip()
        == row2["country"].lower().strip()
    )

    return {
        "name_similarity": name_score,
        "address_similarity": address_score,
        "country_match": country_match
    }


# Example
business1 = {
    "name_normalized": "orelee barbershop",
    "address_normalized": "1795 westchester drive high point nc",
    "country": "US"
}

business2 = {
    "name_normalized": "orelee barbershop",
    "address_normalized": "1795 w westchester dr high point nc",
    "country": "US"
}


features = calculate_features(
    business1,
    business2
)

print("Features:")
print(features)