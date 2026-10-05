from src.model.predict import (
    decision_from_probability,
)


def test_low_risk_application_is_approved():
    assert (
        decision_from_probability(0.10)
        == "DEMO_APPROVE"
    )


def test_medium_risk_application_is_reviewed():
    assert (
        decision_from_probability(0.30)
        == "DEMO_REVIEW"
    )


def test_high_risk_application_is_rejected():
    assert (
        decision_from_probability(0.65)
        == "DEMO_REJECT"
    )


def test_approve_boundary():
    assert (
        decision_from_probability(0.20)
        == "DEMO_REVIEW"
    )


def test_reject_boundary():
    assert (
        decision_from_probability(0.40)
        == "DEMO_REJECT"
    )
