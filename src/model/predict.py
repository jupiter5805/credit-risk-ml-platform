from functools import lru_cache

import joblib
import pandas as pd

from src.config import (
    MODEL_FILE,
    APPROVE_THRESHOLD,
    REVIEW_THRESHOLD,
)

from src.features import (
    engineer_features,
    MODEL_FEATURES,
)


FEATURE_LABELS = {
    "annual_income":
        "Annual income",

    "requested_loan_amount":
        "Requested loan amount",

    "existing_debt":
        "Existing debt",

    "monthly_debt_payments":
        "Monthly debt payments",

    "credit_utilization":
        "Credit utilisation",

    "missed_payments_12m":
        "Missed payments",

    "previous_defaults":
        "Previous default history",

    "credit_history_years":
        "Credit history length",

    "employment_years":
        "Employment stability",

    "recent_credit_enquiries":
        "Recent credit enquiries",

    "loan_term_months":
        "Loan term",

    "debt_to_income":
        "Debt-to-income ratio",

    "loan_to_income":
        "Loan-to-income ratio",
}


@lru_cache(maxsize=1)
def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model not found. "
            "Run python -m src.model.train first."
        )

    return joblib.load(
        MODEL_FILE
    )


def decision_from_probability(
    probability,
):

    if probability < APPROVE_THRESHOLD:
        return "DEMO_APPROVE"

    if probability < REVIEW_THRESHOLD:
        return "DEMO_REVIEW"

    return "DEMO_REJECT"


def explain_prediction(
    pipeline,
    engineered_row,
):

    scaler = pipeline.named_steps[
        "scaler"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    scaled_values = (
        scaler.transform(
            engineered_row[
                MODEL_FEATURES
            ]
        )[0]
    )

    coefficients = (
        model.coef_[0]
    )

    contributions = (
        scaled_values
        * coefficients
    )

    explanations = []

    for feature, contribution in zip(
        MODEL_FEATURES,
        contributions,
    ):

        explanations.append(
            {
                "feature":
                    FEATURE_LABELS.get(
                        feature,
                        feature,
                    ),

                "contribution":
                    round(
                        float(
                            contribution
                        ),
                        4,
                    ),
            }
        )

    risk_drivers = sorted(
        [
            item
            for item in explanations
            if item[
                "contribution"
            ] > 0
        ],
        key=lambda item:
            item[
                "contribution"
            ],
        reverse=True,
    )[:4]

    positive_factors = sorted(
        [
            item
            for item in explanations
            if item[
                "contribution"
            ] < 0
        ],
        key=lambda item:
            item[
                "contribution"
            ],
    )[:4]

    return {
        "risk_drivers":
            risk_drivers,

        "positive_factors":
            positive_factors,
    }


def predict_application(
    application,
):

    pipeline = load_model()

    raw = pd.DataFrame(
        [
            application
        ]
    )

    engineered = (
        engineer_features(
            raw
        )
    )

    probability = float(
        pipeline.predict_proba(
            engineered[
                MODEL_FEATURES
            ]
        )[0, 1]
    )

    decision = (
        decision_from_probability(
            probability
        )
    )

    explanation = (
        explain_prediction(
            pipeline,
            engineered,
        )
    )

    return {
        "predicted_default_probability":
            round(
                probability,
                4,
            ),

        "predicted_default_percentage":
            round(
                probability * 100,
                2,
            ),

        "decision":
            decision,

        "risk_drivers":
            explanation[
                "risk_drivers"
            ],

        "positive_factors":
            explanation[
                "positive_factors"
            ],
    }
