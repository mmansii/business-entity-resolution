from difflib import SequenceMatcher


def similarity(text1, text2):
    return SequenceMatcher(
        None,
        text1.lower(),
        text2.lower()
    ).ratio()


pairs = [
    ("Orelee's Barbershop", "Orelee Barbershop"),
    ("Orelee's Barbershop", "Sweet Barbershop"),
    ("Prime Money", "Prime Money Inc"),
    ("Delta Tetlecommunication Inc", "Delta Telecommunication Inc")
]


for text1, text2 in pairs:
    score = similarity(text1, text2)

    print("--------------------------------")
    print("Text 1:", text1)
    print("Text 2:", text2)
    print("Similarity:", round(score, 3))