import json

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_FILE,
    MODEL_FILE,
    METRICS_FILE,
)

from src.data.generate_data import (
    generate_credit_data,
)

from src.features import (
    engineer_features,
    MODEL_FEATURES,
)


def load_or_create_data():
    if DATA_FILE.exists():
        return pd.read_csv(
            DATA_FILE
        )

    DATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = generate_credit_data()

    df.to_csv(
        DATA_FILE,
        index=False,
    )

    return df


def train_model():
    df = load_or_create_data()

    df = engineer_features(
        df
    )

    X = df[
        MODEL_FEATURES
    ]

    y = df[
        "defaulted"
    ]

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),

        "true_negatives": int(
            matrix[0][0]
        ),

        "false_positives": int(
            matrix[0][1]
        ),

        "false_negatives": int(
            matrix[1][0]
        ),

        "true_positives": int(
            matrix[1][1]
        ),

        "training_rows": int(
            len(X_train)
        ),

        "test_rows": int(
            len(X_test)
        ),

        "default_rate": float(
            y.mean()
        ),
    }

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_FILE,
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        "\nCREDIT RISK MODEL TRAINED"
    )

    print(
        "-------------------------"
    )

    for name, value in metrics.items():

        if isinstance(
            value,
            float,
        ):
            print(
                f"{name}: {value:.4f}"
            )

        else:
            print(
                f"{name}: {value}"
            )

    print(
        f"\nModel saved to: "
        f"{MODEL_FILE}"
    )

    print(
        f"Metrics saved to: "
        f"{METRICS_FILE}"
    )

    return pipeline, metrics


if __name__ == "__main__":
    train_model()
