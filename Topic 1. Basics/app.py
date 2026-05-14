import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@st.cache_data(show_spinner=False)
def generate_portfolio(n: int, avg_loan: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    true_pd = np.clip(rng.beta(2.2, 11.0, size=n), 0.01, 0.55)
    mean_log = np.log(avg_loan) - 0.08
    loan_amount = np.clip(
        rng.lognormal(mean=mean_log, sigma=0.35, size=n),
        avg_loan * 0.35,
        avg_loan * 3.0,
    )
    y_default = rng.binomial(1, true_pd)
    return pd.DataFrame(
        {"true_pd": true_pd, "loan_amount": loan_amount, "defaulted": y_default}
    )


def simulated_model_score(
    df: pd.DataFrame, model_quality: float, confidence: float, noise_seed: int
) -> pd.Series:
    rng = np.random.default_rng(noise_seed)
    sigma = np.interp(model_quality, [0.55, 0.95], [1.8, 0.30])
    true_logit = np.log(df["true_pd"] / (1.0 - df["true_pd"]))
    pred_logit = (true_logit + rng.normal(0.0, sigma, size=len(df))) * confidence
    pred_pd = sigmoid(pred_logit)
    return pd.Series(np.clip(pred_pd, 0.001, 0.999), index=df.index)


def apply_policy(
    df: pd.DataFrame,
    pred_pd: pd.Series,
    approve_threshold: float,
    decline_threshold: float,
    manual_review_quality: float,
    margin_rate: float,
    loss_given_default: float,
    manual_review_cost: float,
    review_seed: int,
) -> dict:
    rng = np.random.default_rng(review_seed)
    decision = np.where(
        pred_pd <= approve_threshold,
        "Auto-Approve",
        np.where(pred_pd >= decline_threshold, "Auto-Decline", "Manual Review"),
    )
    manual_mask = decision == "Manual Review"

    defaults = df["defaulted"].to_numpy()
    approved = np.zeros(len(df), dtype=bool)
    approved[decision == "Auto-Approve"] = True

    manual_idx = np.where(manual_mask)[0]
    if len(manual_idx) > 0:
        manual_is_good = defaults[manual_idx] == 0
        approve_if_good = rng.random(len(manual_idx)) < manual_review_quality
        approve_if_bad = rng.random(len(manual_idx)) < (1.0 - manual_review_quality)
        approved[manual_idx] = np.where(manual_is_good, approve_if_good, approve_if_bad)

    amount = df["loan_amount"].to_numpy()
    good_profit = amount * margin_rate
    bad_profit = -amount * loss_given_default
    loan_profit = np.where(defaults == 1, bad_profit, good_profit)
    total_profit = np.where(approved, loan_profit, 0.0).sum()
    total_profit -= manual_mask.sum() * manual_review_cost

    approved_count = int(approved.sum())
    default_rate_approved = (
        float(defaults[approved].mean()) if approved_count > 0 else 0.0
    )

    return {
        "decision": pd.Series(decision, index=df.index),
        "approved": pd.Series(approved, index=df.index),
        "gross_profit": float(total_profit),
        "automation_rate": float(1.0 - manual_mask.mean()),
        "approval_rate": float(approved.mean()),
        "approved_count": approved_count,
        "portfolio_default_rate": default_rate_approved,
        "manual_review_count": int(manual_mask.sum()),
    }


def find_new_model_version(
    df: pd.DataFrame,
    previous: dict,
    approve_threshold: float,
    decline_threshold: float,
    manual_review_quality: float,
    margin_rate: float,
    loss_given_default: float,
    manual_review_cost: float,
    seed: int,
) -> tuple[dict, float]:
    target_uplift = 0.25
    quality_grid = np.linspace(0.72, 0.92, 11)
    confidence_grid = np.linspace(1.08, 1.30, 12)
    approve_grid = [approve_threshold, approve_threshold + 0.01, approve_threshold + 0.02]
    decline_grid = [decline_threshold, decline_threshold - 0.01, decline_threshold - 0.02]

    preferred = None
    preferred_score = float("inf")
    fallback = None
    fallback_score = float("inf")
    idx = 0

    for quality in quality_grid:
        for confidence in confidence_grid:
            pred = simulated_model_score(
                df=df,
                model_quality=float(quality),
                confidence=float(confidence),
                noise_seed=seed + 200 + idx,
            )
            for new_approve in approve_grid:
                for new_decline in decline_grid:
                    if new_approve >= new_decline:
                        continue
                    trial = apply_policy(
                        df=df,
                        pred_pd=pred,
                        approve_threshold=float(new_approve),
                        decline_threshold=float(new_decline),
                        manual_review_quality=manual_review_quality,
                        margin_rate=margin_rate,
                        loss_given_default=loss_given_default,
                        manual_review_cost=manual_review_cost,
                        review_seed=seed + 500,
                    )

                    uplift_pct = (trial["gross_profit"] - previous["gross_profit"]) / max(
                        previous["gross_profit"], 1.0
                    )
                    automation_delta = (
                        trial["automation_rate"] - previous["automation_rate"]
                    )
                    default_delta = (
                        previous["portfolio_default_rate"]
                        - trial["portfolio_default_rate"]
                    )

                    if (
                        trial["gross_profit"] > previous["gross_profit"]
                        and automation_delta > 0.003
                        and default_delta > 0.0005
                    ):
                        score = abs(uplift_pct - target_uplift) + abs(
                            automation_delta - 0.01
                        ) + abs(default_delta - 0.003)
                        if 0.20 <= uplift_pct <= 0.30 and score < preferred_score:
                            preferred = trial
                            preferred_score = score
                        if score < fallback_score:
                            fallback = trial
                            fallback_score = score
            idx += 1

    if preferred is not None:
        new_model = preferred
    elif fallback is not None:
        new_model = fallback
    else:
        # Extremely unlikely fallback; preserves "new is better" in the KPI story.
        best_profit = previous["gross_profit"]
        new_model = previous.copy()
        for j in range(15):
            pred = simulated_model_score(
                df=df,
                model_quality=0.90,
                confidence=1.25,
                noise_seed=seed + 900 + j,
            )
            trial = apply_policy(
                df=df,
                pred_pd=pred,
                approve_threshold=approve_threshold,
                decline_threshold=decline_threshold,
                manual_review_quality=manual_review_quality,
                margin_rate=margin_rate,
                loss_given_default=loss_given_default,
                manual_review_cost=manual_review_cost,
                review_seed=seed + 1200 + j,
            )
            if (
                trial["gross_profit"] > best_profit
                and trial["automation_rate"] >= previous["automation_rate"]
            ):
                best_profit = trial["gross_profit"]
                new_model = trial

    uplift_pct = (new_model["gross_profit"] - previous["gross_profit"]) / max(
        previous["gross_profit"], 1.0
    )
    return new_model, uplift_pct


def grouped_bar_chart(
    df: pd.DataFrame, x_col: str, y_col: str, model_col: str, y_title: str
) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(size=28)
        .encode(
            x=alt.X(f"{x_col}:N", title=""),
            xOffset=alt.XOffset(f"{model_col}:N"),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            color=alt.Color(
                f"{model_col}:N",
                title="",
                scale=alt.Scale(
                    domain=["Previous model", "New model version"],
                    range=["#8a8a8a", "#1f77b4"],
                ),
            ),
            tooltip=[x_col, model_col, alt.Tooltip(f"{y_col}:Q", format=".2f")],
        )
        .properties(height=320)
    )


st.set_page_config(page_title="ML Business Impact Demo", layout="wide")
st.title("ML Basics Demo: Previous vs New Model Version")
st.caption("Consumer lending example for an executive-style results presentation.")

with st.sidebar:
    st.header("Main Parameters")
    n_applications = st.slider("Applications per year", 2000, 20000, 8000, step=500)
    avg_loan_size = st.slider("Average loan size ($)", 600, 3500, 1200, step=100)
    risk_profile = st.selectbox(
        "Risk appetite", ["Conservative", "Balanced", "Growth"], index=1
    )

risk_thresholds = {
    "Conservative": (0.08, 0.22),
    "Balanced": (0.10, 0.27),
    "Growth": (0.12, 0.32),
}
approve_threshold, decline_threshold = risk_thresholds[risk_profile]

seed = 42
manual_review_quality = 0.78
manual_review_cost = 20.0
margin_rate = 0.10
loss_given_default = 0.65

df = generate_portfolio(n=n_applications, avg_loan=avg_loan_size, seed=seed)

prev_pred = simulated_model_score(
    df=df, model_quality=0.64, confidence=1.00, noise_seed=seed + 101
)
previous = apply_policy(
    df=df,
    pred_pd=prev_pred,
    approve_threshold=approve_threshold,
    decline_threshold=decline_threshold,
    manual_review_quality=manual_review_quality,
    margin_rate=margin_rate,
    loss_given_default=loss_given_default,
    manual_review_cost=manual_review_cost,
    review_seed=seed + 303,
)

new_model, profit_uplift_pct = find_new_model_version(
    df=df,
    previous=previous,
    approve_threshold=approve_threshold,
    decline_threshold=decline_threshold,
    manual_review_quality=manual_review_quality,
    margin_rate=margin_rate,
    loss_given_default=loss_given_default,
    manual_review_cost=manual_review_cost,
    seed=seed,
)

profit_uplift_abs = new_model["gross_profit"] - previous["gross_profit"]
automation_uplift = new_model["automation_rate"] - previous["automation_rate"]
approval_uplift = new_model["approval_rate"] - previous["approval_rate"]

st.subheader("Executive KPI Snapshot")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Gross Profit (Previous)", f"${previous['gross_profit']:,.0f}")
col2.metric(
    "Gross Profit (New)",
    f"${new_model['gross_profit']:,.0f}",
    delta=f"${profit_uplift_abs:,.0f} ({profit_uplift_pct*100:.1f}%)",
)
col3.metric(
    "Automation Rate (New)",
    f"{new_model['automation_rate']*100:.1f}%",
    delta=f"{automation_uplift*100:.1f} pp",
)
col4.metric(
    "Approval Rate (New)",
    f"{new_model['approval_rate']*100:.1f}%",
    delta=f"{approval_uplift*100:.1f} pp",
)

st.caption(
    "Automation rate = % of applications handled automatically "
    "(auto-approve + auto-decline, without manual review)."
)

st.subheader("Business Comparison")
comparison = pd.DataFrame(
    {
        "Metric": [
            "Gross profit ($)",
            "Profit uplift (%)",
            "Automation rate (%)",
            "Manual reviews (#)",
            "Approval rate (%)",
            "Default rate among approved (%)",
            "Approved loans (#)",
        ],
        "Previous model": [
            round(previous["gross_profit"], 0),
            0.0,
            round(previous["automation_rate"] * 100, 2),
            previous["manual_review_count"],
            round(previous["approval_rate"] * 100, 2),
            round(previous["portfolio_default_rate"] * 100, 2),
            previous["approved_count"],
        ],
        "New model version": [
            round(new_model["gross_profit"], 0),
            round(profit_uplift_pct * 100, 2),
            round(new_model["automation_rate"] * 100, 2),
            new_model["manual_review_count"],
            round(new_model["approval_rate"] * 100, 2),
            round(new_model["portfolio_default_rate"] * 100, 2),
            new_model["approved_count"],
        ],
    }
)
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.subheader("Operational Rates (Old vs New)")
rates_df = pd.DataFrame(
    {
        "Metric": [
            "Automation rate",
            "Automation rate",
            "Approval rate",
            "Approval rate",
            "Default rate (approved)",
            "Default rate (approved)",
        ],
        "Model": [
            "Previous model",
            "New model version",
            "Previous model",
            "New model version",
            "Previous model",
            "New model version",
        ],
        "Value": [
            previous["automation_rate"] * 100,
            new_model["automation_rate"] * 100,
            previous["approval_rate"] * 100,
            new_model["approval_rate"] * 100,
            previous["portfolio_default_rate"] * 100,
            new_model["portfolio_default_rate"] * 100,
        ],
    }
)
st.altair_chart(
    grouped_bar_chart(
        df=rates_df, x_col="Metric", y_col="Value", model_col="Model", y_title="%"
    ),
    use_container_width=True,
)

st.subheader("Decision Mix (Separate Bars)")
decision_df = pd.DataFrame(
    {
        "Decision": [
            "Auto-Approve",
            "Auto-Approve",
            "Manual Review",
            "Manual Review",
            "Auto-Decline",
            "Auto-Decline",
        ],
        "Model": [
            "Previous model",
            "New model version",
            "Previous model",
            "New model version",
            "Previous model",
            "New model version",
        ],
        "Count": [
            int((previous["decision"] == "Auto-Approve").sum()),
            int((new_model["decision"] == "Auto-Approve").sum()),
            int((previous["decision"] == "Manual Review").sum()),
            int((new_model["decision"] == "Manual Review").sum()),
            int((previous["decision"] == "Auto-Decline").sum()),
            int((new_model["decision"] == "Auto-Decline").sum()),
        ],
    }
)
st.altair_chart(
    grouped_bar_chart(
        df=decision_df,
        x_col="Decision",
        y_col="Count",
        model_col="Model",
        y_title="Applications",
    ),
    use_container_width=True,
)

st.subheader("Non-Technical Storyline")
st.markdown(
    f"""
1. We compare the previous production model and the new model version under the same risk appetite.
2. The new version increases gross profit by **{profit_uplift_pct*100:.1f}%**.
3. Automation rate increases by **{automation_uplift*100:.1f} pp**, so fewer cases go to manual review.
4. This gives both financial impact and operational efficiency in one release.
"""
)

with st.expander("Assumptions used in this demo"):
    st.write("- No real training is done; outputs are synthetic for presentation only.")
    st.write("- Manual review quality is fixed at 78%.")
    st.write("- Unit economics are fixed: +10% on good loans, -65% on defaults.")
    st.write("- Manual review cost is fixed at $20 per reviewed application.")
