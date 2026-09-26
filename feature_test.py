from difflib import SequenceMatcher


def similarity(text1, text2):
    return SequenceMatcher(
        None,
        str(text1).lower(),
        str(text2).lower()
    ).ratio()


pairs = [
    (
        "Orelee's Barbershop",
        "Orelee Barbershop",
        "1795 Westchester Drive, High Point, NC",
        "1795 W Westchester Dr, High Point, NC"
    ),
    (
        "Orelee's Barbershop",
        "Sweet Barbershop",
        "1795 Westchester Drive, High Point, NC",
        "500 Main Street, Charlotte, NC"
    )
]


for name1, name2, address1, address2 in pairs:

    name_score = similarity(name1, name2)
    address_score = similarity(address1, address2)

    print("--------------------------------")
    print("Name similarity:", round(name_score, 3))
    print("Address similarity:", round(address_score, 3))