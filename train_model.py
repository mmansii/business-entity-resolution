import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# --------------------------------
# LOAD TRAINING DATA
# --------------------------------

data = pd.read_csv("training_data.csv")


# Features
X = data[
    [
        "name_similarity",
        "address_similarity",
        "country_match"
    ]
]

# Target
y = data["label"]


# --------------------------------
# SPLIT DATA
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))


# --------------------------------
# TRAIN MODEL
# --------------------------------

model = LogisticRegression(
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)


# --------------------------------
# TEST MODEL
# --------------------------------

predictions = model.predict(X_test)


print("\nMODEL RESULTS")
print("=" * 60)

print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------
# SHOW MODEL WEIGHTS
# --------------------------------

print("\nMODEL WEIGHTS")
print("=" * 60)

for feature, weight in zip(
    X.columns,
    model.coef_[0]
):

    print(
        feature,
        "->",
        round(weight, 4)
    )


print("\nIntercept:")
print(round(model.intercept_[0], 4))