import streamlit as st

from src.model.predict import predict_application


st.set_page_config(
    page_title="Credit Risk ML Platform",
    page_icon="🏦",
    layout="wide",
)


st.title("🏦 Credit Risk ML Platform")

st.caption(
    "Machine-learning credit risk assessment "
    "using synthetic applicant data."
)

st.info(
    "Portfolio demonstration only. "
    "Not intended for real lending decisions."
)


st.divider()


left, right = st.columns(2)


with left:

    st.subheader("Applicant Finances")

    annual_income = st.number_input(
        "Annual income (£)",
        min_value=1000.0,
        max_value=500000.0,
        value=55000.0,
        step=1000.0,
    )

    requested_loan_amount = st.number_input(
        "Requested loan amount (£)",
        min_value=500.0,
        max_value=250000.0,
        value=8000.0,
        step=500.0,
    )

    existing_debt = st.number_input(
        "Existing debt (£)",
        min_value=0.0,
        max_value=250000.0,
        value=3000.0,
        step=500.0,
    )

    monthly_debt_payments = st.number_input(
        "Monthly debt payments (£)",
        min_value=0.0,
        max_value=10000.0,
        value=250.0,
        step=50.0,
    )

    credit_utilization_percent = st.slider(
        "Credit utilisation (%)",
        min_value=0,
        max_value=100,
        value=20,
    )


with right:

    st.subheader("Credit History")

    missed_payments_12m = st.number_input(
        "Missed payments — last 12 months",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )

    previous_defaults_text = st.selectbox(
        "Previous default?",
        [
            "No",
            "Yes",
        ],
    )

    credit_history_years = st.number_input(
        "Credit history (years)",
        min_value=0.0,
        max_value=50.0,
        value=8.0,
        step=0.5,
    )

    employment_years = st.number_input(
        "Employment stability (years)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
    )

    recent_credit_enquiries = st.number_input(
        "Recent credit enquiries",
        min_value=0,
        max_value=20,
        value=1,
        step=1,
    )

    loan_term_months = st.selectbox(
        "Loan term",
        [
            12,
            24,
            36,
            48,
            60,
        ],
        index=2,
    )


st.divider()


if st.button(
    "Assess Application",
    type="primary",
    use_container_width=True,
):

    application = {
        "annual_income":
            annual_income,

        "requested_loan_amount":
            requested_loan_amount,

        "existing_debt":
            existing_debt,

        "monthly_debt_payments":
            monthly_debt_payments,

        "credit_utilization":
            credit_utilization_percent / 100,

        "missed_payments_12m":
            missed_payments_12m,

        "previous_defaults":
            1
            if previous_defaults_text == "Yes"
            else 0,

        "credit_history_years":
            credit_history_years,

        "employment_years":
            employment_years,

        "recent_credit_enquiries":
            recent_credit_enquiries,

        "loan_term_months":
            loan_term_months,
    }


    result = predict_application(
        application
    )


    probability = result[
        "predicted_default_percentage"
    ]

    decision = result[
        "decision"
    ]


    st.subheader("Assessment Result")


    metric1, metric2 = st.columns(2)


    with metric1:

        st.metric(
            "Predicted Default Risk",
            f"{probability}%"
        )


    with metric2:

        clean_decision = (
            decision
            .replace(
                "DEMO_",
                ""
            )
        )

        st.metric(
            "Recommendation",
            clean_decision
        )


    if decision == "DEMO_APPROVE":

        st.success(
            "✅ DEMO APPROVE"
        )

        st.write(
            "The model estimates that this "
            "application falls within the "
            "lower-risk demonstration range."
        )


    elif decision == "DEMO_REVIEW":

        st.warning(
            "⚠️ DEMO MANUAL REVIEW"
        )

        st.write(
            "The predicted risk is within "
            "the intermediate range and would "
            "require additional review."
        )


    else:

        st.error(
            "❌ DEMO REJECT"
        )

        st.write(
            "The model estimates that this "
            "application falls within the "
            "higher-risk demonstration range."
        )


    st.progress(
        min(
            probability / 100,
            1.0
        )
    )


    st.divider()


    risk_column, positive_column = (
        st.columns(2)
    )


    with risk_column:

        st.subheader(
            "📈 Main Risk Drivers"
        )

        risk_drivers = result[
            "risk_drivers"
        ]

        if risk_drivers:

            for driver in risk_drivers:

                st.write(
                    f"• {driver['feature']}"
                )

        else:

            st.write(
                "No major risk drivers identified."
            )


    with positive_column:

        st.subheader(
            "📉 Risk-Reducing Factors"
        )

        positive_factors = result[
            "positive_factors"
        ]

        if positive_factors:

            for factor in positive_factors:

                st.write(
                    f"• {factor['feature']}"
                )

        else:

            st.write(
                "No major risk-reducing "
                "factors identified."
            )


    st.divider()


    st.subheader(
        "Applicant Summary"
    )


    summary1, summary2, summary3 = (
        st.columns(3)
    )


    monthly_income = (
        annual_income / 12
    )


    debt_to_income = (
        monthly_debt_payments
        / monthly_income
        if monthly_income
        else 0
    )


    loan_to_income = (
        requested_loan_amount
        / annual_income
        if annual_income
        else 0
    )


    with summary1:

        st.metric(
            "Debt-to-Income",
            f"{debt_to_income:.1%}"
        )


    with summary2:

        st.metric(
            "Loan-to-Income",
            f"{loan_to_income:.1%}"
        )


    with summary3:

        st.metric(
            "Credit Utilisation",
            f"{credit_utilization_percent}%"
        )
