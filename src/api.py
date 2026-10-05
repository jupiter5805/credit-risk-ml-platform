from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.model.predict import predict_application


app = FastAPI(
    title="Credit Risk ML Platform",
    description=(
        "Portfolio demonstration API that predicts "
        "simulated credit default risk."
    ),
    version="1.0.0",
)


class CreditApplication(BaseModel):
    annual_income: float = Field(
        gt=0
    )

    requested_loan_amount: float = Field(
        gt=0
    )

    existing_debt: float = Field(
        ge=0
    )

    monthly_debt_payments: float = Field(
        ge=0
    )

    credit_utilization: float = Field(
        ge=0,
        le=1
    )

    missed_payments_12m: int = Field(
        ge=0
    )

    previous_defaults: int = Field(
        ge=0,
        le=1
    )

    credit_history_years: float = Field(
        ge=0
    )

    employment_years: float = Field(
        ge=0
    )

    recent_credit_enquiries: int = Field(
        ge=0
    )

    loan_term_months: int = Field(
        gt=0
    )


@app.get("/")
def root():
    return {
        "project": "Credit Risk ML Platform",
        "status": "running",
        "purpose": "portfolio demonstration",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/assess")
def assess_application(
    application: CreditApplication,
):
    return predict_application(
        application.model_dump()
    )
