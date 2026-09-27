import pandas as pd
import numpy as np


print("=" * 70)
print("TRAINING SIMPLE LOGISTIC REGRESSION MODEL")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "matching_features_v2_labeled.csv"
)

print()
print("Total rows:", len(df))

print(
    "Positive rows:",
    (df["label"] == 1).sum()
)

print(
    "Negative rows:",
    (df["label"] == 0).sum()
)


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    "name_similarity",
    "name_token_similarity",
    "address_similarity",
    "address_token_similarity",
    "address1_present",
    "address2_present",
    "country_match"
]


X = df[feature_columns].fillna(0).astype(float)

y = df["label"].astype(int)


# ============================================================
# SAMPLE NEGATIVES
# ============================================================

positive_df = df[df["label"] == 1].copy()

negative_df = df[df["label"] == 0].copy()


# Use 5 negatives for every positive
negative_sample_size = min(
    len(negative_df),
    len(positive_df) * 5
)


negative_sample = negative_df.sample(
    n=negative_sample_size,
    random_state=42
)


train_df = pd.concat(
    [
        positive_df,
        negative_sample
    ],
    ignore_index=True
)


print()
print("Training rows:", len(train_df))

print(
    "Training positives:",
    (train_df["label"] == 1).sum()
)

print(
    "Training negatives:",
    (train_df["label"] == 0).sum()
)


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

X_train = (
    train_df[feature_columns]
    .fillna(0)
    .astype(float)
    .values
)

y_train = (
    train_df["label"]
    .astype(int)
    .values
)


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

mean = X_train.mean(axis=0)

std = X_train.std(axis=0)

std[std == 0] = 1


X_train_scaled = (
    X_train - mean
) / std


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

weights = np.zeros(
    X_train_scaled.shape[1]
)

bias = 0.0


def sigmoid(z):

    z = np.clip(
        z,
        -50,
        50
    )

    return 1.0 / (
        1.0 + np.exp(-z)
    )


# Learning parameters

learning_rate = 0.05

epochs = 1000


# ============================================================
# TRAIN
# ============================================================

for epoch in range(epochs):

    z = (
        X_train_scaled @ weights
        +
        bias
    )

    predictions = sigmoid(z)


    error = (
        predictions - y_train
    )


    gradient_weights = (
        X_train_scaled.T @ error
        /
        len(y_train)
    )


    gradient_bias = (
        error.mean()
    )


    weights -= (
        learning_rate
        *
        gradient_weights
    )


    bias -= (
        learning_rate
        *
        gradient_bias
    )


    if epoch % 100 == 0:

        loss = -np.mean(
            y_train * np.log(
                predictions + 1e-9
            )
            +
            (1 - y_train)
            *
            np.log(
                1 - predictions + 1e-9
            )
        )

        print(
            f"Epoch {epoch:4d} "
            f"Loss={loss:.6f}"
        )


# ============================================================
# SHOW LEARNED WEIGHTS
# ============================================================

print()
print("=" * 70)
print("LEARNED FEATURE WEIGHTS")
print("=" * 70)


for feature, weight in zip(
    feature_columns,
    weights
):

    print(
        f"{feature:30s} {weight:.6f}"
    )


print()
print(
    f"Bias: {bias:.6f}"
)


# ============================================================
# PREDICT ENTIRE VALIDATION DATA
# ============================================================

X_all = (
    X.values
)

X_all_scaled = (
    X_all - mean
) / std


probabilities = sigmoid(
    X_all_scaled @ weights
    +
    bias
)


df["model_probability"] = probabilities


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    threshold
):

    prediction = (
        df["model_probability"]
        >= threshold
    )

    tp = (
        (df["label"] == 1)
        &
        prediction
    ).sum()

    fp = (
        (df["label"] == 0)
        &
        prediction
    ).sum()

    fn = (
        (df["label"] == 1)
        &
        (~prediction)
    ).sum()


    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )


    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )


    f05 = (
        1.25
        *
        precision
        *
        recall
        /
        (
            0.25
            *
            precision
            +
            recall
        )
        if precision + recall > 0
        else 0
    )


    return (
        tp,
        fp,
        fn,
        precision,
        recall,
        f05
    )


# ============================================================
# TEST THRESHOLDS
# ============================================================

print()
print("=" * 70)
print("MODEL THRESHOLD EXPERIMENT")
print("=" * 70)


for threshold in [
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

    result = evaluate(
        threshold
    )


    print(
        f"Threshold {threshold:.2f} "
        f"TP={result[0]:4d} "
        f"FP={result[1]:4d} "
        f"FN={result[2]:4d} "
        f"P={result[3]:.4f} "
        f"R={result[4]:.4f} "
        f"F0.5={result[5]:.4f}"
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)