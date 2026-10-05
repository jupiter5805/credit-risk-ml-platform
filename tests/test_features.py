import pandas as pd

from src.features import engineer_features


def test_debt_to_income_calculation():
    df = pd.DataFrame(
        [
            {
                "annual_income": 60000,
                "requested_loan_amount": 10000,
                "monthly_debt_payments": 500,
            }
        ]
    )

    result = engineer_features(df)

    assert round(
        result.iloc[0]["debt_to_income"],
        2,
    ) == 0.10


def test_loan_to_income_calculation():
    df = pd.DataFrame(
        [
            {
                "annual_income": 50000,
                "requested_loan_amount": 10000,
                "monthly_debt_payments": 0,
            }
        ]
    )

    result = engineer_features(df)

    assert round(
        result.iloc[0]["loan_to_income"],
        2,
    ) == 0.20


def test_original_dataframe_not_modified():
    df = pd.DataFrame(
        [
            {
                "annual_income": 50000,
                "requested_loan_amount": 10000,
                "monthly_debt_payments": 100,
            }
        ]
    )

    engineer_features(df)

    assert "debt_to_income" not in df.columns
    assert "loan_to_income" not in df.columns
