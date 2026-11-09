import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("../data/sampledcreditcard.csv")

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["Class"].value_counts())


# ============================================================
# 2. PREPROCESSING
# ============================================================

# Convert Time into a time-of-day category
df["SecondsInDay"] = df["Time"] % 86400
df["Hour"] = (df["SecondsInDay"] // 3600).astype(int)


def get_part_of_day(hour):
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 24:
        return "evening"
    else:
        return "night"


df["PartOfDay"] = df["Hour"].apply(get_part_of_day)


# One-hot encode PartOfDay
df = pd.get_dummies(
    df,
    columns=["PartOfDay"],
    drop_first=True
)


# Remove columns that are not directly used as model inputs
X = df.drop(
    ["Class", "Hour", "SecondsInDay"],
    axis=1
)

y = df["Class"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. HANDLE CLASS IMBALANCE
# ============================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# ============================================================
# 6. TRAIN MULTIPLE MODELS
# ============================================================

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        random_state=42
    ),

    "DecisionTree": DecisionTreeClassifier(
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}


results = {}


for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(
        X_train_smote,
        y_train_smote
    )

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("F1-score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    results[name] = {
        "model": model,
        "f1": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# 7. SELECT BEST MODEL
# ============================================================

best_model_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

best_model = results[best_model_name]["model"]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model_name)
print(
    "F1-score:",
    round(results[best_model_name]["f1"], 4)
)


# ============================================================
# 8. SAVE MODEL + PREPROCESSOR
# ============================================================

joblib.dump(
    best_model,
    "model_v2.joblib"
)

joblib.dump(
    scaler,
    "scaler_v2.joblib"
)

joblib.dump(
    list(X.columns),
    "features_v2.joblib"
)


print("\nSaved:")
print("model_v2.joblib")
print("scaler_v2.joblib")
print("features_v2.joblib")