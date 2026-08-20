import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/disease_data.csv"
)

print("\nDataset loaded successfully")

print(
    "Dataset shape:",
    df.shape
)

print("\nFirst five records:")

print(
    df.head()
)


# ==========================================
# 2. REMOVE UNNECESSARY COLUMN
# ==========================================

if "Patient_ID" in df.columns:

    df = df.drop(
        columns=["Patient_ID"]
    )


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")

print(
    df.isnull().sum()
)


# ==========================================
# 4. DEFINE FEATURES AND TARGET
# ==========================================

X = df.drop(
    columns=["Disease"]
)

y = df["Disease"]


# ==========================================
# 5. IDENTIFY DATA TYPES
# ==========================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical Features:")

print(
    categorical_features
)

print("\nNumerical Features:")

print(
    numeric_features
)


# ==========================================
# 6. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ==========================================
# 7. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining records:", len(X_train))

print(
    "Testing records:",
    len(X_test)
)


# ==========================================
# 8. MACHINE LEARNING MODELS
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

    "Naive Bayes":
        GaussianNB(),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        )
}


# ==========================================
# 9. TRAIN MODELS
# ==========================================

results = {}

trained_models = {}


for model_name, model in models.items():

    print(
        "\nTraining:",
        model_name
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results[model_name] = {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1
    }

    trained_models[
        model_name
    ] = pipeline

    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall:",
        round(recall, 4)
    )

    print(
        "F1 Score:",
        round(f1, 4)
    )


# ==========================================
# 10. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(
    results
).T

print("\n==============================")

print("MODEL COMPARISON")

print("==============================")

print(
    results_df
)


# ==========================================
# 11. SELECT BEST MODEL
# ==========================================

best_model_name = results_df[
    "F1 Score"
].idxmax()


best_model = trained_models[
    best_model_name
]


print(
    "\nBest Model:",
    best_model_name
)


# ==========================================
# 12. FINAL PREDICTION
# ==========================================

best_predictions = best_model.predict(
    X_test
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        best_predictions
    )
)


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    best_predictions
)


print(
    "\nConfusion Matrix:"
)

print(
    cm
)


# ==========================================
# 14. SAVE MODEL
# ==========================================

joblib.dump(
    best_model,
    "models/disease_model.pkl"
)


# ==========================================
# 15. SAVE MODEL INFORMATION
# ==========================================

model_information = {

    "best_model":
        best_model_name,

    "results":
        results,

    "features":
        X.columns.tolist(),

    "classes":
        y.unique().tolist()
}


joblib.dump(
    model_information,
    "models/model_info.pkl"
)


print(
    "\nModel saved successfully!"
)

print(
    "models/disease_model.pkl"
)