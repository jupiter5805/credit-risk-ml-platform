import pandas as pd


RAW_FEATURES = [
    "annual_income",
    "requested_loan_amount",
    "existing_debt",
    "monthly_debt_payments",
    "credit_utilization",
    "missed_payments_12m",
    "previous_defaults",
    "credit_history_years",
    "employment_years",
    "recent_credit_enquiries",
    "loan_term_months",
]


MODEL_FEATURES = RAW_FEATURES + [
    "debt_to_income",
    "loan_to_income",
]


def engineer_features(dataframe):
    df = dataframe.copy()

    monthly_income = (
        df["annual_income"]
        .replace(0, pd.NA)
        / 12
    )

    df["debt_to_income"] = (
        df["monthly_debt_payments"]
        / monthly_income
    ).fillna(0)

    df["loan_to_income"] = (
        df["requested_loan_amount"]
        / df["annual_income"].replace(0, pd.NA)
    ).fillna(0)

    df["debt_to_income"] = (
        df["debt_to_income"].clip(0, 5)
    )

    df["loan_to_income"] = (
        df["loan_to_income"].clip(0, 5)
    )

    return df
