def combined_score(name_score, address_score):
    return (
        0.60 * name_score
        + 0.40 * address_score
    )


examples = [
    (0.944, 0.933),
    (0.743, 0.353)
]


for name_score, address_score in examples:

    score = combined_score(
        name_score,
        address_score
    )

    print("--------------------------------")
    print("Name score:", name_score)
    print("Address score:", address_score)
    print("Combined score:", round(score, 3))