import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.model import FEATURE_LABELS
from src.ui import ACCENT, ROOM_COLORS, metrics, style_fig

m = metrics()
mod, base = m["model"], m["baseline"]

st.title("Model performance")
st.markdown(
    f"The model was trained on {m['n_listings_modelled'] - m['test_size']:,} listings and tested on "
    f"**{m['test_size']:,} listings it never saw**. It is compared with a simple baseline: "
    f"*{base['description'].lower()}*."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Typical error (median)", f"{mod['median_abs_pct_error']:.1f}%",
          f"{mod['median_abs_pct_error'] - base['median_abs_pct_error']:+.1f} pts vs baseline", delta_color="inverse")
c2.metric("Average error (MAE)", f"${mod['mae_usd']:.0f}",
          f"{mod['mae_usd'] - base['mae_usd']:+.0f} vs baseline", delta_color="inverse")
c3.metric("R² on log price", f"{mod['r2_log_price']:.2f}",
          f"{mod['r2_log_price'] - base['r2_log_price']:+.2f} vs baseline")
c4.metric("80% range coverage", f"{mod['interval_80_coverage']:.0f}%",
          help="Share of test listings whose real price fell inside the predicted 80% range. Target: 80%.")
st.caption(f"{mod['within_25pct']:.0f}% of test listings were predicted within 25% of their real price. "
           f"The median width of the 80% range is ${mod['interval_80_median_width_usd']:.0f}.")

left, right = st.columns(2)
with left:
    s = pd.DataFrame(m["test_sample"])
    fig = go.Figure()
    for rt, col in ROOM_COLORS.items():
        d = s[s["room_type"] == rt]
        fig.add_trace(go.Scattergl(x=d["actual"], y=d["predicted"], mode="markers", name=rt,
                                   marker=dict(color=col, size=5, opacity=0.5),
                                   hovertemplate=f"{rt}<br>Real: $%{{x:,.0f}}<br>Predicted: $%{{y:,.0f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[10, 1000], y=[10, 1000], mode="lines", name="Perfect prediction",
                             line=dict(color="#898781", dash="dash", width=1.5), hoverinfo="skip"))
    fig.update_layout(title="Predicted vs real price (3,000 test listings)", xaxis_type="log", yaxis_type="log",
                      xaxis_title="Real price ($, log scale)", yaxis_title="Predicted price ($, log scale)")
    fig = style_fig(fig, height=460)
    fig.update_layout(legend=dict(orientation="h", y=-0.22, yanchor="top"), margin_t=40, margin_b=40)
    st.plotly_chart(fig, width="stretch")
    st.caption("Points on the dashed line are perfect predictions. The model is less sure about very cheap "
               "and very expensive listings, which pulls those points toward the middle.")
with right:
    imp = pd.Series(m["feature_importance"]).head(12)
    labels = [FEATURE_LABELS.get(k, k) for k in imp.index]
    fig = go.Figure(go.Bar(y=labels, x=imp.values, orientation="h", marker_color=ACCENT, marker_line_width=0,
                           text=[f"{v:.1f}%" for v in imp.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1f}% of total gain<extra></extra>"))
    fig.update_layout(title="What the model relies on most", yaxis=dict(autorange="reversed"),
                      xaxis_title="Share of importance (gain, %)", showlegend=False)
    st.plotly_chart(style_fig(fig, height=440), width="stretch")
    st.caption("Room type and neighborhood dominate. Bedroom counts parsed from titles are the next most useful signal.")

st.subheader("Accuracy by room type")
br = pd.DataFrame(m["by_room_type"])
st.dataframe(br, hide_index=True, width="stretch", column_config={
    "room_type": "Room type", "n": st.column_config.NumberColumn("Test listings", format="%d"),
    "mae_usd": st.column_config.NumberColumn("Average error", format="$%.0f"),
    "median_abs_pct_error": st.column_config.NumberColumn("Typical error", format="%.1f%%"),
})

st.subheader("Model card")
st.markdown(
    f"""
| | |
|---|---|
| **Algorithm** | LightGBM gradient-boosted trees on log(price). Two LightGBM quantile models (10th and 90th percentile) give the range, widened by conformal calibration (+{m['interval_log_adjustment']:.3f} in log space) to hit 80% coverage. |
| **Training data** | {m['n_listings_modelled']:,} of {m['n_listings_total']:,} NYC listings from July 2019 with prices between $10 and $1,000. |
| **Evaluation** | Random 80/20 split. Interval calibrated on a separate slice of the training split. The "Price vs model" labels in the explorer use 5-fold out-of-fold predictions, so no listing is judged by a model that saw it. |
| **Explanations** | Per-listing TreeSHAP contributions from LightGBM, shown as % effects on the estimator page. |
| **Limitations** | 2019 prices, so it does not reflect inflation or the 2023 NYC short-term rental law. No bedroom, bathroom, amenity or photo data, so size comes only from the title. Prices are nightly list prices, not what guests actually paid. |
| **Retrain** | `python train.py` rebuilds the cleaned data, the model, these metrics and the scored listings in about a minute. |
| **Last trained** | {m['trained_at']} |
"""
)
