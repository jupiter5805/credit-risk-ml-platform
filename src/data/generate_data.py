import numpy as np
import pandas as pd

from src.config import DATA_FILE


def generate_credit_data(
    rows=10000,
    random_state=42,
):
    rng = np.random.default_rng(
        random_state
    )

    annual_income = rng.normal(
        48000,
        18000,
        rows,
    ).clip(15000, 150000)

    requested_loan_amount = rng.normal(
        14000,
        9000,
        rows,
    ).clip(1000, 60000)

    existing_debt = rng.normal(
        9000,
        8500,
        rows,
    ).clip(0, 60000)

    monthly_debt_payments = rng.normal(
        450,
        350,
        rows,
    ).clip(0, 3000)

    credit_utilization = rng.beta(
        2.2,
        3.0,
        rows,
    )

    missed_payments_12m = rng.poisson(
        0.7,
        rows,
    ).clip(0, 10)

    previous_defaults = rng.binomial(
        1,
        0.09,
        rows,
    )

    credit_history_years = rng.normal(
        7,
        4,
        rows,
    ).clip(0, 30)

    employment_years = rng.normal(
        5,
        4,
        rows,
    ).clip(0, 30)

    recent_credit_enquiries = rng.poisson(
        1.5,
        rows,
    ).clip(0, 10)

    loan_term_months = rng.choice(
        [12, 24, 36, 48, 60],
        rows,
        p=[0.10, 0.20, 0.35, 0.20, 0.15],
    )

    monthly_income = (
        annual_income / 12
    )

    debt_to_income = (
        monthly_debt_payments
        / monthly_income
    )

    loan_to_income = (
        requested_loan_amount
        / annual_income
    )

    risk_logit = (
        -4.8
        + 3.0 * debt_to_income
        + 2.3 * credit_utilization
        + 0.32 * missed_payments_12m
        + 1.40 * previous_defaults
        + 1.50 * loan_to_income
        - 0.045 * credit_history_years
        - 0.035 * employment_years
        + 0.11 * recent_credit_enquiries
        + 0.000008 * existing_debt
    )

    default_probability = (
        1
        / (
            1
            + np.exp(-risk_logit)
        )
    )

    defaulted = rng.binomial(
        1,
        default_probability,
    )

    df = pd.DataFrame(
        {
            "annual_income":
                annual_income.round(2),

            "requested_loan_amount":
                requested_loan_amount.round(2),

            "existing_debt":
                existing_debt.round(2),

            "monthly_debt_payments":
                monthly_debt_payments.round(2),

            "credit_utilization":
                credit_utilization.round(4),

            "missed_payments_12m":
                missed_payments_12m,

            "previous_defaults":
                previous_defaults,

            "credit_history_years":
                credit_history_years.round(1),

            "employment_years":
                employment_years.round(1),

            "recent_credit_enquiries":
                recent_credit_enquiries,

            "loan_term_months":
                loan_term_months,

            "defaulted":
                defaulted,
        }
    )

    return df


def main():
    DATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = generate_credit_data()

    df.to_csv(
        DATA_FILE,
        index=False,
    )

    print(
        "\nSynthetic credit dataset created."
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Default rate: "
        f"{df['defaulted'].mean():.2%}"
    )

    print(
        f"Saved to: {DATA_FILE}"
    )


if __name__ == "__main__":
    main()
