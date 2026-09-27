import pandas as pd

df = pd.read_csv("matching_features_v2_labeled_numbers.csv")

# Only pairs with exact address-number agreement
exact = df[df["number_overlap"] >= 1.0].copy()

print("=" * 80)
print("EXACT NUMBER AGREEMENT ANALYSIS")
print("=" * 80)

print(f"\nTotal exact-number pairs: {len(exact):,}")

# ---------------------------------------------------------
# Overall genuine / false counts
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("OVERALL")
print("=" * 80)

print(
    exact["label"]
    .value_counts()
    .rename({0: "False pairs", 1: "Genuine matches"})
    .to_string()
)

# ---------------------------------------------------------
# Address similarity bands
# ---------------------------------------------------------

bins = [
    -0.001,
    0.40,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.001
]

labels = [
    "<.40",
    ".40-.50",
    ".50-.55",
    ".55-.60",
    ".60-.65",
    ".65-.70",
    ".70-.75",
    ".75-.80",
    ".80-.85",
    ".85-.90",
    ".90-.95",
    ".95-1.00"
]

exact["address_band"] = pd.cut(
    exact["address_similarity"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

summary = (
    exact
    .groupby("address_band", observed=False)["label"]
    .agg(
        total="count",
        genuine="sum"
    )
)

summary["false"] = summary["total"] - summary["genuine"]

summary["precision"] = (
    summary["genuine"] / summary["total"]
)

print("\n")
print(summary.to_string())

# ---------------------------------------------------------
# Cumulative thresholds
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("CUMULATIVE THRESHOLD ANALYSIS")
print("=" * 80)

for threshold in [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]:

    subset = exact[
        exact["address_similarity"] >= threshold
    ]

    genuine = (subset["label"] == 1).sum()
    false = (subset["label"] == 0).sum()

    precision = (
        genuine / len(subset)
        if len(subset)
        else 0
    )

    print(
        f"Address >= {threshold:.2f} : "
        f"total={len(subset):5d} "
        f"genuine={genuine:5d} "
        f"false={false:5d} "
        f"precision={precision:.4f}"
    )

# ---------------------------------------------------------
# Exact number + token address similarity
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("EXACT NUMBERS + ADDRESS TOKEN SIMILARITY")
print("=" * 80)

for threshold in [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]:

    subset = exact[
        exact["address_token_similarity"] >= threshold
    ]

    genuine = (subset["label"] == 1).sum()
    false = (subset["label"] == 0).sum()

    precision = (
        genuine / len(subset)
        if len(subset)
        else 0
    )

    print(
        f"Token >= {threshold:.2f} : "
        f"total={len(subset):5d} "
        f"genuine={genuine:5d} "
        f"false={false:5d} "
        f"precision={precision:.4f}"
    )

# ---------------------------------------------------------
# Exact numbers + both address measures
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("COMBINED ADDRESS THRESHOLDS")
print("=" * 80)

tests = [
    (0.50, 0.80),
    (0.55, 0.80),
    (0.60, 0.80),
    (0.65, 0.85),
    (0.70, 0.85),
    (0.60, 0.90),
    (0.65, 0.90),
    (0.70, 0.90),
]

for addr_threshold, token_threshold in tests:

    subset = exact[
        (exact["address_similarity"] >= addr_threshold)
        &
        (exact["address_token_similarity"] >= token_threshold)
    ]

    genuine = (subset["label"] == 1).sum()
    false = (subset["label"] == 0).sum()

    precision = (
        genuine / len(subset)
        if len(subset)
        else 0
    )

    print(
        f"Addr >= {addr_threshold:.2f}, "
        f"Token >= {token_threshold:.2f} : "
        f"total={len(subset):5d} "
        f"genuine={genuine:5d} "
        f"false={false:5d} "
        f"precision={precision:.4f}"
    )

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)